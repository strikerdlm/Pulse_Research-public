"""Tests for GET /experiments/{id}/shap (Phase 6.6)."""
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
        "/experiments", json={"name": "shap", "n_base": n_base, "seed": 42}
    )).json()
    exp_id: str = created["id"]
    assert (await client.post(f"/experiments/{exp_id}/run")).status_code == 202
    for _ in range(100):
        await asyncio.sleep(0.02)
        if (await client.get(f"/experiments/{exp_id}")).json()["status"] == "completed":
            break
    assert (await client.get(f"/experiments/{exp_id}")).json()["status"] == "completed"
    return exp_id


async def test_shap_returns_attribution_for_completed_synthetic_experiment(
    client: httpx.AsyncClient,
) -> None:
    exp_id = await _completed_experiment(client)

    r = await client.get(f"/experiments/{exp_id}/shap")
    assert r.status_code == 200
    body = r.json()
    assert len(body["feature_names"]) == 11
    assert len(body["mean_abs"]) == 11
    assert body["values"] is None
    assert body["seed"] == 42
    assert isinstance(body["base_value"], float)
    # train_mae is XGBoost's training MAE on the design cohort; must be >= 0.
    assert body["train_mae"] >= 0.0


async def test_shap_404_on_unknown_experiment(client: httpx.AsyncClient) -> None:
    r = await client.get(f"/experiments/{uuid.uuid4()}/shap")
    assert r.status_code == 404
    assert r.json()["detail"] == "experiment_not_found"


async def test_shap_409_when_experiment_not_completed(
    client: httpx.AsyncClient,
) -> None:
    created = (await client.post(
        "/experiments", json={"name": "pending", "n_base": 8, "seed": 42}
    )).json()
    r = await client.get(f"/experiments/{created['id']}/shap")
    assert r.status_code == 409
    assert r.json()["detail"] == "experiment_not_completed"


async def test_shap_409_when_no_outputs(
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

    r = await ac.get(f"/experiments/{exp_id}/shap")
    assert r.status_code == 409
    assert r.json()["detail"] == "experiment_has_no_outputs"


async def test_shap_409_when_outputs_contain_nan(
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

    r = await ac.get(f"/experiments/{exp_id}/shap")
    assert r.status_code == 409
    assert r.json()["detail"] == "outputs_contain_nan"


async def test_shap_includes_values_when_requested(
    client: httpx.AsyncClient,
) -> None:
    """?include_samples=true returns the per-row SHAP matrix."""
    exp_id = await _completed_experiment(client)
    r = await client.get(
        f"/experiments/{exp_id}/shap",
        params={"include_samples": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["values"] is not None
    # n_base=8 → 8 * (2*11 + 2) = 192 design rows; 11 features per row.
    assert len(body["values"]) == 192
    assert len(body["values"][0]) == 11


async def test_shap_seed_propagates(
    client: httpx.AsyncClient,
) -> None:
    """Seed is echoed in the response and plumbed to the surrogate fit.

    Note: the synthetic target (_synthetic_row) is a 3-active-feature linear
    function. XGBoost's greedy histogram builder uses no subsampling by default
    (subsample=1.0, colsample_bytree=1.0), so random_state has no effect on
    the resulting trees. Different seeds therefore produce identical mean_abs
    arrays on this dataset — we verify only the echo contract here.
    """
    exp_id = await _completed_experiment(client)
    r1 = await client.get(
        f"/experiments/{exp_id}/shap", params={"seed": 1}
    )
    r2 = await client.get(
        f"/experiments/{exp_id}/shap", params={"seed": 7}
    )
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["seed"] == 1
    assert r2.json()["seed"] == 7
