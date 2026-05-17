"""Paired low-fidelity / high-fidelity run records with parquet I/O.

Each :class:`RunRecord` is one (feature_vector, simulator_output) pair coming
from either the CGEM Fortran core (``Fidelity.LOW``) or the Pulse Physiology
Engine v4.3.1 (``Fidelity.HIGH``). The parquet layout flattens the nested
:class:`FeatureVector19` into ``feat_<axis>`` columns so downstream parquet
consumers (DuckDB, polars, BigQuery export) can filter without unpacking JSON.
"""
from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from pulse_research.schema.features import FeatureVector19


class Fidelity(StrEnum):
    LOW = "low"
    HIGH = "high"


class RunRecord(BaseModel):
    """One simulator-output record. Single-fidelity (orthogonal oracle):
    CGEM rows have ``time_to_gloc_s`` populated and ``cerebral_o2_min``
    None; Pulse rows have the inverse. Paired records are produced
    downstream by joining two RunRecords on ``(run_id, row_index)``.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    fidelity: Fidelity
    features: FeatureVector19
    time_to_gloc_s: float | None = Field(default=None, ge=0.0)
    cerebral_o2_min: float | None = Field(default=None, ge=0.0, le=1.0)
    engine_version: str


def _record_to_row(rec: RunRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": rec.run_id,
        "fidelity": rec.fidelity.value,
        "time_to_gloc_s": rec.time_to_gloc_s,
        "cerebral_o2_min": rec.cerebral_o2_min,
        "engine_version": rec.engine_version,
    }
    for name, value in rec.features.model_dump().items():
        row[f"feat_{name}"] = value
    return row


def write_records_parquet(records: Sequence[RunRecord], path: Path | str) -> None:
    """Write a sequence of :class:`RunRecord` to ``path`` as parquet."""
    df = pd.DataFrame([_record_to_row(r) for r in records])
    df.to_parquet(path, index=False)


def read_records_parquet(path: Path | str) -> pd.DataFrame:
    """Read a parquet file produced by :func:`write_records_parquet`."""
    return pd.read_parquet(path)
