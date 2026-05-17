"""Tests for the SSE event stream."""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from pulse_research.api.app import create_app
from pulse_research.api.events import EventBroker
from pulse_research.api.runners import SyntheticRunner
from pulse_research.api.store import InMemoryStore


@pytest.fixture()
async def client_pieces() -> AsyncIterator[
    tuple[httpx.AsyncClient, InMemoryStore, EventBroker]
]:
    store = InMemoryStore()
    broker = EventBroker()
    # Slow the runner so the SSE subscriber sees progress events live.
    runner = SyntheticRunner(
        store=store,
        broker=broker,
        batch_size=16,
        sleep_between_batches=0.01,
    )
    app = create_app(store=store, broker=broker, runner=runner)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac, store, broker


def _parse_sse(text: str) -> list[dict[str, object]]:
    """Naive SSE parser: every 'data: <json>' line becomes one event dict."""
    events: list[dict[str, object]] = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[6:].strip()
            if not payload or payload == "{}":
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                continue
    return events


async def test_sse_unknown_experiment_returns_404(
    client_pieces: tuple[httpx.AsyncClient, InMemoryStore, EventBroker],
) -> None:
    client, _, _ = client_pieces
    r = await client.get(f"/experiments/{uuid.uuid4()}/events")
    assert r.status_code == 404


async def test_sse_emits_completion_event(
    client_pieces: tuple[httpx.AsyncClient, InMemoryStore, EventBroker],
) -> None:
    """Subscribe before running; expect at least one event with status=completed."""
    client, _, _ = client_pieces
    created = (await client.post(
        "/experiments", json={"name": "sse1", "n_base": 4, "seed": 42}
    )).json()
    exp_id = created["id"]

    async def consume_stream() -> str:
        chunks: list[str] = []
        async with client.stream(
            "GET", f"/experiments/{exp_id}/events", timeout=10.0
        ) as response:
            assert response.status_code == 200
            async for chunk in response.aiter_text():
                chunks.append(chunk)
        return "".join(chunks)

    consumer_task = asyncio.create_task(consume_stream())
    # Give the subscriber a beat to attach before the runner publishes.
    await asyncio.sleep(0.05)
    r = await client.post(f"/experiments/{exp_id}/run")
    assert r.status_code == 202

    text = await asyncio.wait_for(consumer_task, timeout=10.0)
    events = _parse_sse(text)
    assert any(e.get("status") == "completed" for e in events), events


async def test_sse_on_already_completed_yields_terminal_event(
    client_pieces: tuple[httpx.AsyncClient, InMemoryStore, EventBroker],
) -> None:
    """Connecting after completion still emits the terminal status, then closes."""
    client, _, _ = client_pieces
    created = (await client.post(
        "/experiments", json={"name": "sse2", "n_base": 4, "seed": 42}
    )).json()
    exp_id = created["id"]
    await client.post(f"/experiments/{exp_id}/run")
    # Wait for completion.
    for _ in range(100):
        await asyncio.sleep(0.02)
        if (await client.get(f"/experiments/{exp_id}")).json()["status"] == "completed":
            break

    async with client.stream(
        "GET", f"/experiments/{exp_id}/events", timeout=5.0
    ) as response:
        assert response.status_code == 200
        text = ""
        async for chunk in response.aiter_text():
            text += chunk
    events = _parse_sse(text)
    assert events
    assert events[-1].get("status") == "completed"
