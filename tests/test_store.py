"""Tests for the in-memory experiment store."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

import numpy as np

from pulse_research.api.models import ExperimentStatus
from pulse_research.api.store import Experiment, InMemoryStore
from pulse_research.sensitivity.sobol_design import build_design


def _design() -> np.ndarray:
    X, _ = build_design(n_base=4, seed=42)
    return X


def test_create_assigns_uuid_and_timestamp() -> None:
    store = InMemoryStore()
    exp = store.create(name="run1", n_base=4, seed=42, design=_design())
    assert isinstance(exp, Experiment)
    assert isinstance(exp.id, UUID)
    assert isinstance(exp.created_at, datetime)
    assert exp.status is ExperimentStatus.PENDING
    assert exp.progress == 0.0
    assert exp.outputs is None


def test_get_returns_created_experiment() -> None:
    store = InMemoryStore()
    exp = store.create(name="run1", n_base=4, seed=42, design=_design())
    assert store.get(exp.id) is exp


def test_get_unknown_returns_none() -> None:
    import uuid

    store = InMemoryStore()
    assert store.get(uuid.uuid4()) is None


def test_list_returns_all() -> None:
    store = InMemoryStore()
    store.create(name="a", n_base=4, seed=42, design=_design())
    store.create(name="b", n_base=4, seed=43, design=_design())
    items = store.list()
    assert {e.name for e in items} == {"a", "b"}


def test_separate_stores_dont_share_state() -> None:
    s1 = InMemoryStore()
    s2 = InMemoryStore()
    s1.create(name="a", n_base=4, seed=42, design=_design())
    assert s2.list() == []
