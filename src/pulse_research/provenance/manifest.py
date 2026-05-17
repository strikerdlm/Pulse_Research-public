"""Provenance manifest schema + helpers for Phase 7.1b paired runs.

The :class:`ProvenanceManifest` is the single source of truth for "what
produced this parquet". The audit script in ``scripts/audit_run_records.py``
reads it back and verifies that every CGEM row in the parquet is
recomputable from a re-trained surrogate, and that the training data file
hash matches the one recorded here. Hard rule from the Phase 7.1b spec:
no field in the parquet may exist without a corresponding live model call,
and the manifest records what that call was.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DesignManifest(BaseModel):
    """Captures the Saltelli design that drove both arms."""

    model_config = ConfigDict(extra="forbid")

    axis_names: list[str]
    n_rows: int
    saltelli_calc_second_order: bool


class CgemArmManifest(BaseModel):
    """Captures the CGEM-surrogate arm: training data, fit info, timings."""

    model_config = ConfigDict(extra="forbid")

    training_parquet_path: str
    training_parquet_sha256: str
    fit_info: dict[str, object]
    # "predict_expected_time_array" (P(event) * E[time|event=1]) was the
    # original Phase 7.1a channel, retired 2026-05-16 after Phase 7.3 ROR
    # diagnostic. "predict_array" (regressor-only conditional time
    # E[time|event=1]) is the post-2026-05-16 channel for direct WF2013
    # LOCINDTI comparability above the +4.7 Gz event threshold.
    output_channel: Literal["predict_expected_time_array", "predict_array"]
    row_count: int
    error_count: int
    wall_clock_s: float
    started_at: str  # ISO-8601
    finished_at: str  # ISO-8601


class PulseArmManifest(BaseModel):
    """Captures the Pulse-Docker arm: image digest, timings, error rate."""

    model_config = ConfigDict(extra="forbid")

    docker_image: str
    docker_image_digest: str
    row_count: int
    error_count: int
    wall_clock_s: float
    started_at: str  # ISO-8601
    finished_at: str  # ISO-8601


class ProvenanceManifest(BaseModel):
    """Top-level manifest written next to the production parquet."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal["7.1b"]
    schema_version: Literal[1] = 1
    seed: int
    n_pulse_base: int
    n_cgem_base: int
    runtime: dict[str, str]  # python_version, xgboost_version, hostname
    design: DesignManifest
    cgem: CgemArmManifest
    pulse: PulseArmManifest


_CHUNK = 65536


def compute_parquet_sha256(path: Path | str) -> str:
    """Stream ``path`` through SHA-256 and return the hex digest.

    Streaming so we don't hold a multi-MB parquet in memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(m: ProvenanceManifest, path: Path | str) -> None:
    """Serialize ``m`` to ``path`` as indented JSON (creates parents)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(m.model_dump_json(indent=2))
