"""In-memory experiment store.

A pluggable boundary: the same interface can be backed by SQLite, Postgres, or
DuckDB in later phases. Phase 4 MVP keeps everything in process memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from uuid import UUID, uuid4

import numpy as np

from pulse_research.api.models import ExperimentStatus


@dataclass
class Experiment:
    """Mutable internal experiment record; API responses use the pydantic
    ``ExperimentDetail`` / ``ExperimentSummary`` projections."""

    id: UUID
    name: str
    n_base: int
    seed: int
    design: np.ndarray
    created_at: datetime
    status: ExperimentStatus = ExperimentStatus.PENDING
    progress: float = 0.0
    error: str | None = None
    outputs: np.ndarray | None = field(default=None, repr=False)
    engine_label: str = ""
    failed_rows: int = 0


class InMemoryStore:
    """Thread-safe dict-backed store keyed by experiment UUID."""

    def __init__(self) -> None:
        self._experiments: dict[UUID, Experiment] = {}
        self._lock = RLock()

    def create(
        self,
        name: str,
        n_base: int,
        seed: int,
        design: np.ndarray,
    ) -> Experiment:
        with self._lock:
            exp = Experiment(
                id=uuid4(),
                name=name,
                n_base=n_base,
                seed=seed,
                design=design,
                created_at=datetime.now(UTC),
            )
            self._experiments[exp.id] = exp
            return exp

    def get(self, experiment_id: UUID) -> Experiment | None:
        with self._lock:
            return self._experiments.get(experiment_id)

    def list(self) -> list[Experiment]:
        with self._lock:
            return list(self._experiments.values())
