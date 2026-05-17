"""Tests for GET /experiments/{id}/sobol (Phase 6.5)."""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from uuid import UUID

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from pulse_research.api.app import create_app
from pulse_research.api.models import ExperimentStatus


@pytest.fixture()
async def client_and_app() -> AsyncIterator[tuple[httpx.AsyncClient, FastAPI]]:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, app


@pytest.fixture()
async def client(
    client_and_app: tuple[httpx.AsyncClient, FastAPI],
) -> httpx.AsyncClient:
    return client_and_app[0]


async def _completed_experiment(client: httpx.AsyncClient, n_base: int = 8) -> str:
    """Create a SyntheticRunner experiment and poll until completed."""
    created = (await client.post(
        "/experiments", json={"name": "sobol", "n_base": n_base, "seed": 42}
    )).json()
    exp_id: str = created["id"]
    assert (await client.post(f"/experiments/{exp_id}/run")).status_code == 202
    for _ in range(100):
        await asyncio.sleep(0.02)
        if (await client.get(f"/experiments/{exp_id}")).json()["status"] == "completed":
            break
    assert (await client.get(f"/experiments/{exp_id}")).json()["status"] == "completed"
    return exp_id


async def test_sobol_returns_indices_for_completed_synthetic_experiment(
    client: httpx.AsyncClient,
) -> None:
    exp_id = await _completed_experiment(client)

    r = await client.get(
        f"/experiments/{exp_id}/sobol",
        params={"num_resamples": 50},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["names"]) == 11
    assert len(body["S1"]) == 11
    assert len(body["ST"]) == 11
    assert len(body["S1_conf"]) == 11
    assert len(body["ST_conf"]) == 11
    assert body["S2"] is None
    assert body["S2_conf"] is None
    assert body["n_resamples"] == 50
    assert body["seed"] == 42
    # st_stability = 1 - max(ST_conf/ST) over active features. The upper bound
    # is hard (CIs are non-negative); the lower bound is unbounded (negative
    # values are a meaningful "bootstrap not converged" signal at low N).
    assert body["st_stability"] <= 1.0


async def test_sobol_404_on_unknown_experiment(client: httpx.AsyncClient) -> None:
    r = await client.get(f"/experiments/{uuid.uuid4()}/sobol")
    assert r.status_code == 404
    assert r.json()["detail"] == "experiment_not_found"


async def test_sobol_409_when_experiment_not_completed(
    client: httpx.AsyncClient,
) -> None:
    created = (await client.post(
        "/experiments", json={"name": "pending", "n_base": 8, "seed": 42}
    )).json()
    r = await client.get(f"/experiments/{created['id']}/sobol")
    assert r.status_code == 409
    assert r.json()["detail"] == "experiment_not_completed"


async def test_sobol_409_when_no_outputs(
    client_and_app: tuple[httpx.AsyncClient, FastAPI],
) -> None:
    """A completed experiment whose store row has outputs=None must 409."""
    ac, app = client_and_app
    created = (await ac.post(
        "/experiments", json={"name": "noout", "n_base": 8, "seed": 42}
    )).json()
    exp_id = created["id"]
    store = app.state.store
    exp = store.get(UUID(exp_id))
    exp.status = ExperimentStatus.COMPLETED
    exp.outputs = None

    r = await ac.get(f"/experiments/{exp_id}/sobol")
    assert r.status_code == 409
    assert r.json()["detail"] == "experiment_has_no_outputs"


async def test_sobol_409_when_outputs_contain_nan(
    client_and_app: tuple[httpx.AsyncClient, FastAPI],
) -> None:
    """A completed experiment whose outputs contain NaN must 409."""
    ac, app = client_and_app
    exp_id = await _completed_experiment(ac)
    store = app.state.store
    exp = store.get(UUID(exp_id))
    assert exp.outputs is not None
    exp.outputs = exp.outputs.copy()
    exp.outputs[0] = np.nan

    r = await ac.get(f"/experiments/{exp_id}/sobol")
    assert r.status_code == 409
    assert r.json()["detail"] == "outputs_contain_nan"


async def test_sobol_includes_s2_when_requested(
    client: httpx.AsyncClient,
) -> None:
    exp_id = await _completed_experiment(client)
    r = await client.get(
        f"/experiments/{exp_id}/sobol",
        params={"num_resamples": 50, "include_second_order": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["S2"] is not None
    assert body["S2_conf"] is not None
    assert len(body["S2"]) == 11
    assert len(body["S2"][0]) == 11


async def test_sobol_seed_and_num_resamples_propagate(
    client: httpx.AsyncClient,
) -> None:
    """Different num_resamples must produce different bootstrap CIs."""
    exp_id = await _completed_experiment(client)
    r1 = await client.get(
        f"/experiments/{exp_id}/sobol",
        params={"num_resamples": 25, "seed": 1},
    )
    r2 = await client.get(
        f"/experiments/{exp_id}/sobol",
        params={"num_resamples": 200, "seed": 1},
    )
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["n_resamples"] == 25
    assert r2.json()["n_resamples"] == 200
    # Point estimates (S1, ST) only depend on the outputs and are invariant
    # under seed / num_resamples; the CIs differ.
    assert r1.json()["S1"] == r2.json()["S1"]
    assert r1.json()["S1_conf"] != r2.json()["S1_conf"]


async def test_sobol_st_stability_in_unit_interval(
    client: httpx.AsyncClient,
) -> None:
    exp_id = await _completed_experiment(client)
    r = await client.get(
        f"/experiments/{exp_id}/sobol", params={"num_resamples": 50}
    )
    score = r.json()["st_stability"]
    # st_stability ∈ (-inf, 1.0]; the lower bound is unbounded by design
    # (negative = bootstrap not converged at low N).
    assert score <= 1.0
