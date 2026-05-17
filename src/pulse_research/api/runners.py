"""Experiment runners.

Two implementations ship out of the box, both built on the same
:class:`_BatchedRowRunner` spine:

* :class:`SyntheticRunner` — closed-form toy outputs from each design row;
  the CI default; first runnable end-to-end smoke target.
* :class:`CGEMRunner` — calls ``cgem_wrapper.run_cgem_centrifuge`` once per
  design row via :mod:`pulse_research.api.cgem_glue`. Used in production
  when ``CGEM_ROOT`` is set; tests inject a fake ``row_fn``.

Both runners publish ``running`` status events as they batch through the
design matrix and a single ``completed`` (or ``failed``) terminal event.
Per-row failures DO NOT fail the experiment; failed rows are recorded as
``np.nan`` in ``outputs`` and counted in ``Experiment.failed_rows``.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID

import numpy as np

from pulse_research.api.events import EventBroker
from pulse_research.api.models import ExperimentStatus
from pulse_research.api.store import InMemoryStore

RowFn = Callable[[np.ndarray], "RowOutput"]

RunnerKind = Literal["synthetic", "cgem", "pulse"]
AVAILABLE_RUNNER_KINDS: tuple[RunnerKind, ...] = ("synthetic", "cgem", "pulse")
RUNNER_ENV_VAR = "PULSE_RESEARCH_RUNNER"
_DEFAULT_RUNNER_KIND: RunnerKind = "synthetic"


@dataclass
class RowOutput:
    """Per-row simulator output.

    ``time_to_gloc_s`` is the only field consumed by the surrogate today; the
    optional ``cerebral_o2_min`` is reserved for the Phase 4.7 Pulse runner
    that will fill it. ``error`` carries the failure cause when both floats
    are ``None``.
    """

    time_to_gloc_s: float | None
    cerebral_o2_min: float | None = None
    error: str | None = None


def _synthetic_row(row: np.ndarray) -> RowOutput:
    """Toy ``time_to_gloc_s`` shaped so tests can assert non-trivial variation.

    Tolerance decreases with Gz peak and onset rate; increases with FiO2
    above the 0.21 baseline. Not physically meaningful — placeholder until
    the CGEM and Pulse runners take over.
    """
    gz_peak = float(row[0])
    gz_rate = float(row[1])
    fio2 = float(row[9])
    raw = 20.0 - 1.5 * gz_peak - 0.3 * gz_rate + 5.0 * (fio2 - 0.21)
    return RowOutput(time_to_gloc_s=max(0.0, raw))


class Runner(Protocol):
    """A runner consumes an experiment ID and mutates the store as it works."""

    async def run(self, experiment_id: UUID) -> None: ...


class _BatchedRowRunner:
    """Shared per-row iteration spine.

    Subclasses fix the ``row_fn`` and the ``engine_label``; the base class
    owns batching, event publishing, output array assembly, and terminal
    status transitions. Row failures degrade gracefully: a ``RowOutput``
    whose ``time_to_gloc_s`` is ``None`` writes ``np.nan`` into the output
    column and increments ``Experiment.failed_rows``.
    """

    def __init__(
        self,
        store: InMemoryStore,
        broker: EventBroker,
        row_fn: RowFn,
        *,
        engine_label: str,
        batch_size: int = 64,
        sleep_between_batches: float = 0.0,
    ) -> None:
        self.store = store
        self.broker = broker
        self.row_fn = row_fn
        self.engine_label = engine_label
        self.batch_size = batch_size
        self.sleep_between_batches = sleep_between_batches

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    async def _eval_row(self, row: np.ndarray) -> RowOutput:
        """Run ``row_fn`` in a worker thread so the event loop stays responsive."""
        return await asyncio.to_thread(self.row_fn, row)

    async def run(self, experiment_id: UUID) -> None:
        exp = self.store.get(experiment_id)
        if exp is None:
            return

        exp.status = ExperimentStatus.RUNNING
        exp.progress = 0.0
        exp.engine_label = self.engine_label
        exp.failed_rows = 0
        await self.broker.publish(
            experiment_id,
            {"status": "running", "progress": 0.0, "ts": self._now()},
        )

        try:
            n = exp.design.shape[0]
            outputs = np.full(n, np.nan, dtype=float)
            failed = 0
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                for i in range(start, end):
                    result = await self._eval_row(exp.design[i])
                    if result.time_to_gloc_s is None:
                        failed += 1
                    else:
                        outputs[i] = result.time_to_gloc_s
                exp.failed_rows = failed
                exp.progress = end / n
                await self.broker.publish(
                    experiment_id,
                    {
                        "status": "running",
                        "progress": exp.progress,
                        "ts": self._now(),
                    },
                )
                if self.sleep_between_batches > 0.0:
                    await asyncio.sleep(self.sleep_between_batches)
                else:
                    await asyncio.sleep(0)
            exp.outputs = outputs
            exp.status = ExperimentStatus.COMPLETED
            exp.progress = 1.0
            await self.broker.publish(
                experiment_id,
                {"status": "completed", "progress": 1.0, "ts": self._now()},
            )
        except Exception as e:  # pragma: no cover - runner-wide failure
            exp.status = ExperimentStatus.FAILED
            exp.error = str(e)
            await self.broker.publish(
                experiment_id,
                {
                    "status": "failed",
                    "progress": exp.progress,
                    "ts": self._now(),
                    "error": str(e),
                },
            )


class SyntheticRunner(_BatchedRowRunner):
    """Closed-form runner; Phase 4 default."""

    def __init__(
        self,
        store: InMemoryStore,
        broker: EventBroker,
        *,
        batch_size: int = 256,
        sleep_between_batches: float = 0.0,
    ) -> None:
        super().__init__(
            store,
            broker,
            _synthetic_row,
            engine_label="synthetic",
            batch_size=batch_size,
            sleep_between_batches=sleep_between_batches,
        )


class CGEMRunner(_BatchedRowRunner):
    """Per-row CGEM Fortran execution via ``run_cgem_centrifuge``.

    The wrapper module is resolved at construction time:
      1. Use the ``row_fn`` argument if provided (tests inject fakes).
      2. Otherwise call :func:`pulse_research.api.cgem_glue.make_cgem_row_fn`
         which uses the ``CGEM_ROOT`` env var to locate the upstream wrapper.
    """

    def __init__(
        self,
        store: InMemoryStore,
        broker: EventBroker,
        *,
        row_fn: RowFn | None = None,
        batch_size: int = 64,
        sleep_between_batches: float = 0.0,
    ) -> None:
        if row_fn is None:
            from pulse_research.api.cgem_glue import make_cgem_row_fn

            row_fn = make_cgem_row_fn()
        super().__init__(
            store,
            broker,
            row_fn,
            engine_label="cgem",
            batch_size=batch_size,
            sleep_between_batches=sleep_between_batches,
        )


class PulseRunner(_BatchedRowRunner):
    """Per-row Pulse Physiology Engine v4.3.1 execution via the
    ``pulse-ds:4.3.1`` Docker image.

    Default ``batch_size`` is small because Pulse calls are seconds-long
    (Docker startup + engine stabilization + simulation advance + CSV write).
    Tests inject a fake ``row_fn`` so CI never invokes ``docker run``.
    """

    def __init__(
        self,
        store: InMemoryStore,
        broker: EventBroker,
        *,
        row_fn: RowFn | None = None,
        batch_size: int = 16,
        sleep_between_batches: float = 0.0,
    ) -> None:
        if row_fn is None:
            from pulse_research.api.pulse_glue import make_pulse_row_fn

            row_fn = make_pulse_row_fn()
        super().__init__(
            store,
            broker,
            row_fn,
            engine_label="pulse",
            batch_size=batch_size,
            sleep_between_batches=sleep_between_batches,
        )


def resolve_runner_kind(env: Mapping[str, str] | None = None) -> RunnerKind:
    """Read ``PULSE_RESEARCH_RUNNER`` from the environment.

    Defaults to ``"synthetic"`` when the variable is unset or empty.
    Lowercases and strips whitespace before comparing. Raises
    :class:`ValueError` when the value is non-empty but not one of the
    supported kinds — operators should see misconfig at startup, not at
    first runner invocation.
    """
    if env is None:
        env = os.environ
    raw = (env.get(RUNNER_ENV_VAR) or "").strip().lower()
    if not raw:
        return _DEFAULT_RUNNER_KIND
    if raw not in AVAILABLE_RUNNER_KINDS:
        allowed = ", ".join(repr(k) for k in AVAILABLE_RUNNER_KINDS)
        raise ValueError(
            f"{RUNNER_ENV_VAR}={raw!r} is not a valid runner kind; expected {allowed}"
        )
    return raw


def make_runner(
    kind: RunnerKind,
    store: InMemoryStore,
    broker: EventBroker,
) -> Runner:
    """Build a runner of the requested kind.

    The :class:`CGEMRunner` branch may raise ``RuntimeError`` from
    :func:`pulse_research.api.cgem_glue.make_cgem_row_fn` if ``CGEM_ROOT`` is
    not configured — by design, we fail at startup so the misconfig is
    visible to the operator.
    """
    if kind == "synthetic":
        return SyntheticRunner(store=store, broker=broker)
    if kind == "cgem":
        return CGEMRunner(store=store, broker=broker)
    if kind == "pulse":
        return PulseRunner(store=store, broker=broker)
    allowed = ", ".join(repr(k) for k in AVAILABLE_RUNNER_KINDS)
    raise ValueError(f"unknown runner kind {kind!r}; expected {allowed}")
