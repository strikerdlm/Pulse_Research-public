"""Tests for the SyntheticRunner and CGEMRunner."""
from __future__ import annotations

import asyncio

import numpy as np

from pulse_research.api.events import EventBroker
from pulse_research.api.models import ExperimentStatus
from pulse_research.api.runners import (
    CGEMRunner,
    PulseRunner,
    RowOutput,
    SyntheticRunner,
)
from pulse_research.api.store import InMemoryStore
from pulse_research.sensitivity.sobol_design import build_design


def _design() -> np.ndarray:
    X, _ = build_design(n_base=4, seed=42)
    return X


async def test_runner_completes_synthetic_experiment() -> None:
    store = InMemoryStore()
    broker = EventBroker()
    runner = SyntheticRunner(store=store, broker=broker, batch_size=32)
    exp = store.create(name="x", n_base=4, seed=42, design=_design())

    await runner.run(exp.id)
    exp_after = store.get(exp.id)
    assert exp_after is not None
    assert exp_after.status is ExperimentStatus.COMPLETED
    assert exp_after.progress == 1.0
    assert exp_after.outputs is not None
    assert exp_after.outputs.shape == (exp.design.shape[0],)


async def test_runner_publishes_running_then_completed() -> None:
    store = InMemoryStore()
    broker = EventBroker()
    runner = SyntheticRunner(store=store, broker=broker, batch_size=32)
    exp = store.create(name="x", n_base=4, seed=42, design=_design())
    q = broker.subscribe(exp.id)

    await runner.run(exp.id)

    statuses: list[str] = []
    while not q.empty():
        ev = await q.get()
        statuses.append(ev["status"])
    assert statuses[0] == "running"
    assert statuses[-1] == "completed"


async def test_runner_no_op_on_unknown_experiment() -> None:
    import uuid

    store = InMemoryStore()
    broker = EventBroker()
    runner = SyntheticRunner(store=store, broker=broker)
    # Should not raise; the runner returns silently.
    await runner.run(uuid.uuid4())


async def test_runner_outputs_are_deterministic() -> None:
    store_a = InMemoryStore()
    store_b = InMemoryStore()
    broker = EventBroker()
    runner_a = SyntheticRunner(store=store_a, broker=broker)
    runner_b = SyntheticRunner(store=store_b, broker=broker)

    design = _design()
    a = store_a.create(name="a", n_base=4, seed=42, design=design)
    b = store_b.create(name="b", n_base=4, seed=42, design=design.copy())

    await asyncio.gather(runner_a.run(a.id), runner_b.run(b.id))
    after_a = store_a.get(a.id)
    after_b = store_b.get(b.id)
    assert after_a is not None and after_a.outputs is not None
    assert after_b is not None and after_b.outputs is not None
    np.testing.assert_array_equal(after_a.outputs, after_b.outputs)


async def test_synthetic_runner_sets_engine_label() -> None:
    store = InMemoryStore()
    broker = EventBroker()
    runner = SyntheticRunner(store=store, broker=broker)
    exp = store.create(name="x", n_base=4, seed=42, design=_design())
    await runner.run(exp.id)
    after = store.get(exp.id)
    assert after is not None
    assert after.engine_label == "synthetic"
    assert after.failed_rows == 0


async def test_cgem_runner_uses_injected_row_fn_and_completes() -> None:
    calls: list[np.ndarray] = []

    def fake_row_fn(row: np.ndarray) -> RowOutput:
        calls.append(row.copy())
        return RowOutput(time_to_gloc_s=10.0, cerebral_o2_min=None)

    store = InMemoryStore()
    broker = EventBroker()
    runner = CGEMRunner(store=store, broker=broker, row_fn=fake_row_fn, batch_size=32)
    exp = store.create(name="cgem-test", n_base=4, seed=42, design=_design())

    await runner.run(exp.id)
    after = store.get(exp.id)
    assert after is not None
    assert after.status is ExperimentStatus.COMPLETED
    assert after.engine_label == "cgem"
    assert after.failed_rows == 0
    assert len(calls) == exp.design.shape[0]
    assert after.outputs is not None
    assert (after.outputs == 10.0).all()


async def test_cgem_runner_tolerates_row_failures() -> None:
    counter = {"i": 0}

    def flaky_row_fn(row: np.ndarray) -> RowOutput:
        counter["i"] += 1
        if counter["i"] % 3 == 0:
            return RowOutput(
                time_to_gloc_s=None,
                cerebral_o2_min=None,
                error="simulated subprocess fail",
            )
        return RowOutput(time_to_gloc_s=12.0, cerebral_o2_min=None)

    store = InMemoryStore()
    broker = EventBroker()
    runner = CGEMRunner(store=store, broker=broker, row_fn=flaky_row_fn, batch_size=8)
    exp = store.create(name="cgem-flaky", n_base=4, seed=42, design=_design())

    await runner.run(exp.id)
    after = store.get(exp.id)
    assert after is not None
    assert after.status is ExperimentStatus.COMPLETED
    assert after.failed_rows > 0
    assert after.outputs is not None
    assert np.isnan(after.outputs).any()
    successful = ~np.isnan(after.outputs)
    assert (after.outputs[successful] == 12.0).all()


async def test_pulse_runner_uses_injected_row_fn_and_completes() -> None:
    """PulseRunner with an injected fake row_fn must complete and tag the
    experiment with engine_label='pulse'. No Docker invoked."""
    calls: list[np.ndarray] = []

    def fake_row_fn(row: np.ndarray) -> RowOutput:
        calls.append(row.copy())
        return RowOutput(time_to_gloc_s=None, cerebral_o2_min=0.85)

    store = InMemoryStore()
    broker = EventBroker()
    runner = PulseRunner(store=store, broker=broker, row_fn=fake_row_fn, batch_size=8)
    exp = store.create(name="pulse-test", n_base=4, seed=42, design=_design())

    await runner.run(exp.id)
    after = store.get(exp.id)
    assert after is not None
    assert after.status is ExperimentStatus.COMPLETED
    assert after.engine_label == "pulse"
    assert after.failed_rows == exp.design.shape[0]  # all None time_to_gloc_s
    assert len(calls) == exp.design.shape[0]
    # outputs stay NaN because Pulse does not produce time_to_gloc_s
    assert after.outputs is not None
    assert np.isnan(after.outputs).all()
