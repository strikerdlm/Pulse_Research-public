"""Tests for the provenance manifest (Phase 7.1b)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pulse_research.provenance.manifest import (
    CgemArmManifest,
    DesignManifest,
    ProvenanceManifest,
    PulseArmManifest,
    compute_parquet_sha256,
    write_manifest,
)


def _example_manifest() -> ProvenanceManifest:
    return ProvenanceManifest(
        phase="7.1b",
        seed=42,
        n_pulse_base=128,
        n_cgem_base=128,
        runtime={
            "python_version": "3.12.3",
            "xgboost_version": "2.1.4",
            "hostname": "test-host",
        },
        design=DesignManifest(
            axis_names=["gz_peak", "gz_onset_rate"],
            n_rows=3072,
            saltelli_calc_second_order=True,
        ),
        cgem=CgemArmManifest(
            training_parquet_path="/x.parquet",
            training_parquet_sha256="abc123",
            fit_info={"n_train": 3240, "n_train_event": 205},
            output_channel="predict_expected_time_array",
            row_count=3072,
            error_count=0,
            wall_clock_s=0.5,
            started_at="2026-05-16T00:00:00",
            finished_at="2026-05-16T00:00:01",
        ),
        pulse=PulseArmManifest(
            docker_image="pulse-ds:4.3.1",
            docker_image_digest="sha256:abc",
            row_count=3072,
            error_count=3,
            wall_clock_s=18000.0,
            started_at="2026-05-16T00:00:02",
            finished_at="2026-05-16T05:00:02",
        ),
    )


def test_compute_parquet_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    payload = b"hello phase 7.1b"
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert compute_parquet_sha256(p) == expected


def test_compute_parquet_sha256_streams_large_file(tmp_path: Path) -> None:
    """The helper must stream; assert correctness on a >64 KiB file."""
    p = tmp_path / "big.bin"
    payload = b"x" * (1024 * 1024 + 17)  # 1 MiB + tail
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert compute_parquet_sha256(p) == expected


def test_provenance_manifest_round_trips(tmp_path: Path) -> None:
    m = _example_manifest()
    out = tmp_path / "manifest.json"
    write_manifest(m, out)
    payload = json.loads(out.read_text())
    assert payload["phase"] == "7.1b"
    assert payload["pulse"]["error_count"] == 3
    assert payload["cgem"]["output_channel"] == "predict_expected_time_array"
    m2 = ProvenanceManifest.model_validate(payload)
    assert m2 == m


def test_provenance_manifest_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError):
        ProvenanceManifest(
            phase="9.9",  # type: ignore[arg-type]  # not allowed by Literal
            seed=42,
            n_pulse_base=128,
            n_cgem_base=128,
            runtime={},
            design=DesignManifest(
                axis_names=[], n_rows=0, saltelli_calc_second_order=True
            ),
            cgem=CgemArmManifest(
                training_parquet_path="",
                training_parquet_sha256="",
                fit_info={},
                output_channel="predict_expected_time_array",
                row_count=0,
                error_count=0,
                wall_clock_s=0.0,
                started_at="",
                finished_at="",
            ),
            pulse=PulseArmManifest(
                docker_image="",
                docker_image_digest="",
                row_count=0,
                error_count=0,
                wall_clock_s=0.0,
                started_at="",
                finished_at="",
            ),
        )


def test_provenance_manifest_rejects_unknown_output_channel() -> None:
    """Channel-switched 2026-05-16: both predict_expected_time_array (P*E)
    and predict_array (regressor-only conditional time) are now valid.
    Test asserts that an unknown channel name is still rejected.
    """
    with pytest.raises(ValueError):
        CgemArmManifest(
            training_parquet_path="",
            training_parquet_sha256="",
            fit_info={},
            output_channel="predict_garbage",  # type: ignore[arg-type]  # unknown: rejected
            row_count=0,
            error_count=0,
            wall_clock_s=0.0,
            started_at="",
            finished_at="",
        )


def test_write_manifest_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "subdir" / "manifest.json"
    write_manifest(_example_manifest(), out)
    assert out.is_file()
    payload = json.loads(out.read_text())
    assert payload["seed"] == 42
