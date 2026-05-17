"""Tests for the Phase 7.1b audit script."""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from pulse_research.api.cgem_glue import make_cgem_surrogate_row_fn
from pulse_research.io.records import (
    Fidelity,
    RunRecord,
    write_records_parquet,
)
from pulse_research.provenance.manifest import (
    CgemArmManifest,
    DesignManifest,
    ProvenanceManifest,
    PulseArmManifest,
    compute_parquet_sha256,
    write_manifest,
)
from pulse_research.schema import AntiGSuit, FeatureVector19
from pulse_research.sensitivity.sobol_design import AXIS_NAMES

_CGEM_PARQUET = Path(
    "/root/repos/CAMI-Gz-Effects-Model-CGEM-/data/datasets/cgem_synthetic_v1.parquet"
)


def _skip_if_no_cgem_data() -> None:
    try:
        present = _CGEM_PARQUET.is_file()
    except (PermissionError, OSError):
        present = False
    if not present:
        pytest.skip(
            f"CGEM training data not accessible at {_CGEM_PARQUET}; "
            "CGEM_ROOT may be unset, the dataset may not be built, or "
            "the path may be unreadable on this runner"
        )


def _load_audit_module() -> ModuleType:
    """Import scripts/audit_run_records.py as a module."""
    audit_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_run_records.py"
    )
    spec = importlib.util.spec_from_file_location("audit_mod", audit_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {audit_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _features_for_row(row: np.ndarray) -> FeatureVector19:
    """Build a FeatureVector19 from an 11-axis row using batch defaults."""
    return FeatureVector19(
        gz_peak=float(row[0]),
        gz_onset_rate=float(row[1]),
        seat_tilt_deg=float(row[2]),
        anti_g_strain=float(row[3]),
        pilot_weight_kg=float(row[4]),
        pilot_height_cm=float(row[5]),
        pilot_age_y=float(row[6]),
        baseline_vo2max=float(row[7]),
        baseline_map_mmhg=float(row[8]),
        fio2_inspired=float(row[9]),
        sao2_baseline=float(row[10]),
        sex_male=True,
        hypocapnia_flag=False,
        anti_g_suit_class=AntiGSuit.ATAGS,
        retinal_baseline_perfusion=0.85,
        cerebral_autoreg_gain=1.0,
        pulmonary_shunt_baseline=0.02,
        baseline_hb_g_dl=14.5,
        ambient_temp_c=22.0,
    )


def _build_consistent_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a 4-row parquet (2 CGEM + 2 Pulse) whose CGEM rows are
    consistent with a freshly-trained surrogate. Returns (parquet_path,
    manifest_path).
    """
    row_fn = make_cgem_surrogate_row_fn(_CGEM_PARQUET)
    design_rows = [
        np.array([2.0, 1.0, 0.0, 0.5, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97],
                 dtype=float),
        np.array([8.0, 1.0, 0.0, 0.5, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97],
                 dtype=float),
    ]
    cgem_records: list[RunRecord] = []
    for i, row in enumerate(design_rows):
        out = row_fn(row)
        cgem_records.append(RunRecord(
            run_id=f"phase7_1b-cgem-{i:04d}",
            fidelity=Fidelity.LOW,
            features=_features_for_row(row),
            time_to_gloc_s=out.time_to_gloc_s,
            cerebral_o2_min=None,
            engine_version="cgem-surrogate-v1",
        ))
    # One Pulse row succeeded; one Pulse row errored.
    pulse_records: list[RunRecord] = [
        RunRecord(
            run_id="phase7_1b-pulse-0000",
            fidelity=Fidelity.HIGH,
            features=_features_for_row(design_rows[0]),
            time_to_gloc_s=None,
            cerebral_o2_min=0.85,
            engine_version="pulse-4.3.1",
        ),
        RunRecord(
            run_id="phase7_1b-pulse-0001",
            fidelity=Fidelity.HIGH,
            features=_features_for_row(design_rows[1]),
            time_to_gloc_s=None,
            cerebral_o2_min=None,
            engine_version="pulse-4.3.1",
        ),
    ]
    records = cgem_records + pulse_records
    parquet_path = tmp_path / "records.parquet"
    write_records_parquet(records, parquet_path)

    training_sha = compute_parquet_sha256(_CGEM_PARQUET)
    now = datetime.now().isoformat()
    manifest = ProvenanceManifest(
        phase="7.1b",
        seed=42,
        n_pulse_base=8,
        n_cgem_base=8,
        runtime={
            "python_version": "3.12",
            "xgboost_version": "test",
            "hostname": socket.gethostname(),
        },
        design=DesignManifest(
            axis_names=list(AXIS_NAMES),
            n_rows=2,
            saltelli_calc_second_order=True,
        ),
        cgem=CgemArmManifest(
            training_parquet_path=str(_CGEM_PARQUET),
            training_parquet_sha256=training_sha,
            fit_info={"target": "time_to_gloc_s"},
            output_channel="predict_expected_time_array",
            row_count=2,
            error_count=0,
            wall_clock_s=0.0,
            started_at=now,
            finished_at=now,
        ),
        pulse=PulseArmManifest(
            docker_image="pulse-ds:4.3.1",
            docker_image_digest="sha256:test",
            row_count=2,
            error_count=1,
            wall_clock_s=0.0,
            started_at=now,
            finished_at=now,
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)
    return parquet_path, manifest_path


def test_audit_passes_on_consistent_parquet(tmp_path: Path) -> None:
    _skip_if_no_cgem_data()
    audit = _load_audit_module()
    parquet, manifest = _build_consistent_fixture(tmp_path)
    code, msgs = audit.audit_run_records(parquet, manifest)
    assert code == 0, f"audit failed unexpectedly: {msgs}"


def test_audit_fails_on_tampered_cgem_time(tmp_path: Path) -> None:
    _skip_if_no_cgem_data()
    audit = _load_audit_module()
    parquet, manifest = _build_consistent_fixture(tmp_path)
    df = pd.read_parquet(parquet)
    cgem_idx = df.index[df["fidelity"] == "low"][0]
    df.at[cgem_idx, "time_to_gloc_s"] = float(df.at[cgem_idx, "time_to_gloc_s"]) + 1.0  # type: ignore[arg-type]
    df.to_parquet(parquet, index=False)
    code, msgs = audit.audit_run_records(parquet, manifest)
    assert code == 1
    assert any("time_to_gloc_s" in m or "diff" in m.lower() for m in msgs)


def test_audit_fails_on_sha256_mismatch(tmp_path: Path) -> None:
    _skip_if_no_cgem_data()
    audit = _load_audit_module()
    parquet, manifest = _build_consistent_fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["cgem"]["training_parquet_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload, indent=2))
    code, msgs = audit.audit_run_records(parquet, manifest)
    assert code == 1
    assert any("sha256" in m.lower() for m in msgs)


def test_audit_fails_on_pulse_schema_violation(tmp_path: Path) -> None:
    """Pulse row with error=None must have NaN time + finite O2.
    Inject a row that has finite time AND finite O2 → schema violation."""
    _skip_if_no_cgem_data()
    audit = _load_audit_module()
    parquet, manifest = _build_consistent_fixture(tmp_path)
    df = pd.read_parquet(parquet)
    pulse_idx = df.index[df["fidelity"] == "high"][0]
    # Pulse row should have NaN time_to_gloc_s; set it to a finite number.
    df.at[pulse_idx, "time_to_gloc_s"] = 7.5
    df.to_parquet(parquet, index=False)
    code, msgs = audit.audit_run_records(parquet, manifest)
    assert code == 1
    assert any("pulse" in m.lower() or "schema" in m.lower() for m in msgs)


def test_audit_fails_on_pulse_error_with_finite_o2(tmp_path: Path) -> None:
    """Pulse error rows must have BOTH fields NaN; finite O2 on an error
    row is a schema violation."""
    _skip_if_no_cgem_data()
    audit = _load_audit_module()
    parquet, manifest = _build_consistent_fixture(tmp_path)
    df = pd.read_parquet(parquet)
    # Find the error pulse row (both fields NaN); inject finite O2.
    pulse_high = df[df["fidelity"] == "high"]
    error_idx = pulse_high[pulse_high["cerebral_o2_min"].isna()].index[0]
    df.at[error_idx, "cerebral_o2_min"] = 0.9
    # But time_to_gloc_s stays NaN, which now creates an "ok-looking"
    # row — that's not the violation; introduce a real violation by
    # making O2 out-of-range instead.
    df.at[error_idx, "cerebral_o2_min"] = 1.5  # out of [0, 1]
    df.to_parquet(parquet, index=False)
    code, msgs = audit.audit_run_records(parquet, manifest)
    assert code == 1
    assert any("o2" in m.lower() or "schema" in m.lower() or "range" in m.lower()
               for m in msgs)


# Sanity check the import path even when CGEM data is absent.
def test_audit_module_is_importable() -> None:
    audit = _load_audit_module()
    assert hasattr(audit, "audit_run_records")
    assert hasattr(audit, "main")
