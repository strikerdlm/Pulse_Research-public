"""Tests for the runner factory and env-var reader."""
from __future__ import annotations

import pytest

from pulse_research.api.events import EventBroker
from pulse_research.api.runners import (
    AVAILABLE_RUNNER_KINDS,
    RUNNER_ENV_VAR,
    CGEMRunner,
    PulseRunner,
    SyntheticRunner,
    make_runner,
    resolve_runner_kind,
)
from pulse_research.api.store import InMemoryStore


def test_resolve_runner_kind_default_is_synthetic() -> None:
    assert resolve_runner_kind(env={}) == "synthetic"


def test_resolve_runner_kind_empty_string_is_synthetic() -> None:
    assert resolve_runner_kind(env={RUNNER_ENV_VAR: ""}) == "synthetic"


def test_resolve_runner_kind_explicit_synthetic() -> None:
    assert resolve_runner_kind(env={RUNNER_ENV_VAR: "synthetic"}) == "synthetic"


def test_resolve_runner_kind_cgem() -> None:
    assert resolve_runner_kind(env={RUNNER_ENV_VAR: "cgem"}) == "cgem"


def test_resolve_runner_kind_pulse() -> None:
    assert resolve_runner_kind(env={RUNNER_ENV_VAR: "pulse"}) == "pulse"


def test_resolve_runner_kind_case_and_whitespace() -> None:
    assert resolve_runner_kind(env={RUNNER_ENV_VAR: "  CGEM  "}) == "cgem"
    assert resolve_runner_kind(env={RUNNER_ENV_VAR: "Synthetic"}) == "synthetic"


def test_resolve_runner_kind_invalid_raises() -> None:
    with pytest.raises(ValueError, match="not a valid runner kind"):
        resolve_runner_kind(env={RUNNER_ENV_VAR: "garbage"})


def test_make_runner_synthetic() -> None:
    store, broker = InMemoryStore(), EventBroker()
    runner = make_runner("synthetic", store, broker)
    assert isinstance(runner, SyntheticRunner)
    assert runner.engine_label == "synthetic"


def test_make_runner_cgem_uses_make_cgem_row_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_runner('cgem', ...) constructs CGEMRunner; we monkeypatch the row
    factory so no real CGEM is needed."""
    sentinel_calls: list[str] = []

    def fake_make_cgem_row_fn() -> object:
        sentinel_calls.append("called")
        return lambda row: None

    monkeypatch.setattr(
        "pulse_research.api.cgem_glue.make_cgem_row_fn",
        fake_make_cgem_row_fn,
    )

    store, broker = InMemoryStore(), EventBroker()
    runner = make_runner("cgem", store, broker)
    assert isinstance(runner, CGEMRunner)
    assert runner.engine_label == "cgem"
    assert sentinel_calls == ["called"]


def test_make_runner_pulse_uses_make_pulse_row_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_runner('pulse', ...) constructs PulseRunner; we monkeypatch the
    row factory so no Docker is invoked."""
    sentinel_calls: list[str] = []

    def fake_make_pulse_row_fn() -> object:
        sentinel_calls.append("called")
        return lambda row: None

    monkeypatch.setattr(
        "pulse_research.api.pulse_glue.make_pulse_row_fn",
        fake_make_pulse_row_fn,
    )

    store, broker = InMemoryStore(), EventBroker()
    runner = make_runner("pulse", store, broker)
    assert isinstance(runner, PulseRunner)
    assert runner.engine_label == "pulse"
    assert sentinel_calls == ["called"]


def test_make_runner_unknown_kind_raises() -> None:
    store, broker = InMemoryStore(), EventBroker()
    with pytest.raises(ValueError, match="unknown runner kind"):
        make_runner("garbage", store, broker)  # type: ignore[arg-type]


def test_available_runner_kinds_locked() -> None:
    assert AVAILABLE_RUNNER_KINDS == ("synthetic", "cgem", "pulse")
