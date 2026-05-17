"""Tests for the operational introspection endpoint and env-var path."""
from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from pulse_research.api.app import create_app
from pulse_research.api.cgem_glue import CGEM_ENV_VAR
from pulse_research.api.pulse_glue import (
    DEFAULT_PULSE_IMAGE,
    PULSE_IMAGE_ENV_VAR,
    PULSE_WORK_DIR_ENV_VAR,
)
from pulse_research.api.runners import RUNNER_ENV_VAR


@pytest.fixture()
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


async def test_get_runner_default_synthetic(client: httpx.AsyncClient) -> None:
    r = await client.get("/runner")
    assert r.status_code == 200
    body = r.json()
    assert body["active_kind"] == "synthetic"
    assert body["engine_label"] == "synthetic"
    assert body["available_kinds"] == ["synthetic", "cgem", "pulse"]
    assert "cgem" in body
    assert "pulse" in body
    assert body["pulse"]["image"] == DEFAULT_PULSE_IMAGE


async def test_get_runner_reports_cgem_root_state(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CGEM_ENV_VAR, "/tmp/fake-cgem-root")
    r = await client.get("/runner")
    body = r.json()
    assert body["cgem"]["configured"] is True
    assert body["cgem"]["root"] == "/tmp/fake-cgem-root"


async def test_get_runner_reports_cgem_unset_state(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(CGEM_ENV_VAR, raising=False)
    r = await client.get("/runner")
    body = r.json()
    assert body["cgem"]["configured"] is False
    assert body["cgem"]["root"] is None


async def test_create_app_env_var_path_picks_cgem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting PULSE_RESEARCH_RUNNER=cgem causes create_app to wire CGEMRunner.

    We monkeypatch make_cgem_row_fn so no real CGEM is needed; this exercises
    the env-var → resolve_runner_kind → make_runner → CGEMRunner code path
    end-to-end inside create_app.
    """
    monkeypatch.setenv(RUNNER_ENV_VAR, "cgem")
    monkeypatch.setattr(
        "pulse_research.api.cgem_glue.make_cgem_row_fn",
        lambda: (lambda row: None),
    )

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/runner")
    body = r.json()
    assert body["active_kind"] == "cgem"
    assert body["engine_label"] == "cgem"


async def test_create_app_invalid_env_var_raises_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RUNNER_ENV_VAR, "bogus-runner")
    with pytest.raises(ValueError, match="not a valid runner kind"):
        create_app()


async def test_get_runner_reports_pulse_image_env_override(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PULSE_IMAGE_ENV_VAR, "pulse-ds:nightly")
    monkeypatch.setenv(PULSE_WORK_DIR_ENV_VAR, "/var/pulse-work")
    r = await client.get("/runner")
    body = r.json()
    assert body["pulse"]["image"] == "pulse-ds:nightly"
    assert body["pulse"]["work_dir"] == "/var/pulse-work"


async def test_create_app_env_var_path_picks_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PULSE_RESEARCH_RUNNER=pulse causes create_app to wire PulseRunner.

    We monkeypatch make_pulse_row_fn so no Docker is invoked.
    """
    monkeypatch.setenv(RUNNER_ENV_VAR, "pulse")
    monkeypatch.setattr(
        "pulse_research.api.pulse_glue.make_pulse_row_fn",
        lambda: (lambda row: None),
    )

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/runner")
    body = r.json()
    assert body["active_kind"] == "pulse"
    assert body["engine_label"] == "pulse"
