"""Tests for the FastAPI experiment endpoints."""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import httpx
import pytest

from pulse_research.api.app import create_app


@pytest.fixture()
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


async def test_health_returns_ok(client: httpx.AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


async def test_create_experiment_201(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/experiments",
        json={"name": "run1", "n_base": 4, "seed": 42},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "run1"
    assert body["n_base"] == 4
    assert body["seed"] == 42
    assert body["status"] == "pending"
    assert body["n_design_rows"] == 4 * (2 * 11 + 2)
    uuid.UUID(body["id"])  # parses cleanly


async def test_create_non_power_of_two_returns_422(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/experiments",
        json={"name": "bad", "n_base": 1000, "seed": 42},
    )
    assert r.status_code == 422


async def test_list_returns_created(client: httpx.AsyncClient) -> None:
    await client.post("/experiments", json={"name": "a", "n_base": 4, "seed": 42})
    await client.post("/experiments", json={"name": "b", "n_base": 4, "seed": 43})
    r = await client.get("/experiments")
    assert r.status_code == 200
    names = {item["name"] for item in r.json()}
    assert names == {"a", "b"}


async def test_get_unknown_returns_404(client: httpx.AsyncClient) -> None:
    r = await client.get(f"/experiments/{uuid.uuid4()}")
    assert r.status_code == 404


async def test_get_existing_returns_detail(client: httpx.AsyncClient) -> None:
    created = (await client.post(
        "/experiments", json={"name": "z", "n_base": 4, "seed": 42}
    )).json()
    r = await client.get(f"/experiments/{created['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == created["id"]
    assert body["has_outputs"] is False
    assert body["progress"] == 0.0
    assert body["error"] is None


async def test_run_unknown_returns_404(client: httpx.AsyncClient) -> None:
    r = await client.post(f"/experiments/{uuid.uuid4()}/run")
    assert r.status_code == 404


async def test_run_kicks_off_and_reaches_completed(client: httpx.AsyncClient) -> None:
    created = (await client.post(
        "/experiments", json={"name": "go", "n_base": 4, "seed": 42}
    )).json()
    exp_id = created["id"]

    r = await client.post(f"/experiments/{exp_id}/run")
    assert r.status_code == 202

    # Wait for the background task to finish (n=96 rows, trivial work).
    for _ in range(50):
        await asyncio.sleep(0.02)
        detail = (await client.get(f"/experiments/{exp_id}")).json()
        if detail["status"] in ("completed", "failed"):
            break
    assert detail["status"] == "completed"
    assert detail["progress"] == 1.0
    assert detail["has_outputs"] is True


async def test_double_run_returns_409(client: httpx.AsyncClient) -> None:
    created = (await client.post(
        "/experiments", json={"name": "dup", "n_base": 4, "seed": 42}
    )).json()
    exp_id = created["id"]

    r1 = await client.post(f"/experiments/{exp_id}/run")
    assert r1.status_code == 202

    # Wait for terminal state, then a second /run must be rejected.
    for _ in range(50):
        await asyncio.sleep(0.02)
        if (await client.get(f"/experiments/{exp_id}")).json()["status"] == "completed":
            break

    r2 = await client.post(f"/experiments/{exp_id}/run")
    assert r2.status_code == 409


async def test_create_rejects_empty_name(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/experiments", json={"name": "", "n_base": 4, "seed": 42}
    )
    assert r.status_code == 422


async def test_data_endpoint_unknown_returns_404(client: httpx.AsyncClient) -> None:
    r = await client.get(f"/experiments/{uuid.uuid4()}/data")
    assert r.status_code == 404


async def test_data_endpoint_pending_returns_design_no_outputs(
    client: httpx.AsyncClient,
) -> None:
    created = (await client.post(
        "/experiments", json={"name": "pending", "n_base": 4, "seed": 42}
    )).json()
    r = await client.get(f"/experiments/{created['id']}/data")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert len(body["axes"]) == 11
    assert body["n_design_rows"] == 96
    assert body["n_returned"] == 96  # default sample=500 clamped to 96
    assert len(body["rows"]) == body["n_returned"]
    assert all(len(row) == 11 for row in body["rows"])
    assert all(v is None for v in body["outputs"])
    assert body["output_range"] == {"min": None, "max": None}


async def test_data_endpoint_sample_param_clamps_low(
    client: httpx.AsyncClient,
) -> None:
    created = (await client.post(
        "/experiments", json={"name": "s10", "n_base": 4, "seed": 42}
    )).json()
    r = await client.get(f"/experiments/{created['id']}/data?sample=10")
    body = r.json()
    assert body["n_returned"] == 10
    assert len(body["rows"]) == 10


async def test_data_endpoint_sample_deterministic(
    client: httpx.AsyncClient,
) -> None:
    created = (await client.post(
        "/experiments", json={"name": "det", "n_base": 8, "seed": 42}
    )).json()
    r1 = (await client.get(f"/experiments/{created['id']}/data?sample=16")).json()
    r2 = (await client.get(f"/experiments/{created['id']}/data?sample=16")).json()
    assert r1["rows"] == r2["rows"]


async def test_data_endpoint_after_completed_has_outputs(
    client: httpx.AsyncClient,
) -> None:
    created = (await client.post(
        "/experiments", json={"name": "done", "n_base": 4, "seed": 42}
    )).json()
    exp_id = created["id"]
    await client.post(f"/experiments/{exp_id}/run")
    for _ in range(50):
        await asyncio.sleep(0.02)
        if (await client.get(f"/experiments/{exp_id}")).json()["status"] == "completed":
            break

    r = await client.get(f"/experiments/{exp_id}/data?sample=20")
    body = r.json()
    assert body["status"] == "completed"
    assert body["n_returned"] == 20
    assert all(isinstance(v, (int, float)) for v in body["outputs"])
    assert body["output_range"]["min"] is not None
    assert body["output_range"]["max"] is not None
    assert body["output_range"]["min"] <= body["output_range"]["max"]
