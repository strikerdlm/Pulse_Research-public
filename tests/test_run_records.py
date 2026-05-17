"""Tests for paired low-fid / high-fid run records and parquet I/O."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from pulse_research.io.records import (
    Fidelity,
    RunRecord,
    read_records_parquet,
    write_records_parquet,
)
from pulse_research.schema.features import AntiGSuit, FeatureVector19


def _make_record(fidelity: Fidelity, gloc_s: float) -> RunRecord:
    return RunRecord(
        run_id=f"r-{fidelity.value}-{gloc_s:.0f}",
        fidelity=fidelity,
        features=FeatureVector19(
            gz_peak=6.0,
            gz_onset_rate=3.0,
            seat_tilt_deg=15.0,
            anti_g_strain=0.5,
            pilot_weight_kg=80.0,
            pilot_height_cm=175.0,
            pilot_age_y=35.0,
            baseline_vo2max=50.0,
            baseline_map_mmhg=90.0,
            fio2_inspired=0.21,
            sao2_baseline=0.97,
            sex_male=True,
            hypocapnia_flag=False,
            anti_g_suit_class=AntiGSuit.ATAGS,
            retinal_baseline_perfusion=0.85,
            cerebral_autoreg_gain=1.0,
            pulmonary_shunt_baseline=0.02,
            baseline_hb_g_dl=14.5,
            ambient_temp_c=22.0,
        ),
        time_to_gloc_s=gloc_s,
        cerebral_o2_min=0.55,
        engine_version=(
            "cgem-fortran-2026.05" if fidelity is Fidelity.LOW else "pulse-4.3.1"
        ),
    )


def test_parquet_roundtrip_preserves_records(tmp_path: Path) -> None:
    records = [
        _make_record(Fidelity.LOW, 12.0),
        _make_record(Fidelity.HIGH, 11.5),
    ]
    path = tmp_path / "records.parquet"
    write_records_parquet(records, path)
    assert path.exists()
    df = read_records_parquet(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["fidelity"]) == {"low", "high"}
    assert df.loc[df["fidelity"] == "low", "time_to_gloc_s"].iloc[0] == 12.0


def test_parquet_columns_flatten_features(tmp_path: Path) -> None:
    records = [_make_record(Fidelity.LOW, 12.0)]
    path = tmp_path / "records.parquet"
    write_records_parquet(records, path)
    df = read_records_parquet(path)
    assert "feat_gz_peak" in df.columns
    assert "feat_fio2_inspired" in df.columns
    assert "feat_anti_g_suit_class" in df.columns
    assert "run_id" in df.columns
    assert "fidelity" in df.columns
    assert "engine_version" in df.columns


def test_invalid_record_rejected_before_write(tmp_path: Path) -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunRecord(
            run_id="bad",
            fidelity=Fidelity.LOW,
            features=FeatureVector19(
                gz_peak=6.0, gz_onset_rate=3.0, seat_tilt_deg=15.0,
                anti_g_strain=0.5, pilot_weight_kg=80.0,
                pilot_height_cm=175.0, pilot_age_y=35.0,
                baseline_vo2max=50.0, baseline_map_mmhg=90.0,
                fio2_inspired=0.21, sao2_baseline=0.97,
                sex_male=True, hypocapnia_flag=False,
                anti_g_suit_class=AntiGSuit.ATAGS,
                retinal_baseline_perfusion=0.85, cerebral_autoreg_gain=1.0,
                pulmonary_shunt_baseline=0.02, baseline_hb_g_dl=14.5,
                ambient_temp_c=22.0,
            ),
            time_to_gloc_s=-1.0,  # invalid
            cerebral_o2_min=0.55,
            engine_version="pulse-4.3.1",
        )


def test_engine_version_propagates(tmp_path: Path) -> None:
    records = [
        _make_record(Fidelity.LOW, 10.0),
        _make_record(Fidelity.HIGH, 9.5),
    ]
    path = tmp_path / "records.parquet"
    write_records_parquet(records, path)
    df = read_records_parquet(path)
    assert "cgem-fortran-2026.05" in df["engine_version"].tolist()
    assert "pulse-4.3.1" in df["engine_version"].tolist()


# ---------------------------------------------------------------------------
# Optional-field tests (orthogonal-oracle schema)
# ---------------------------------------------------------------------------

def _baseline_features() -> FeatureVector19:
    return FeatureVector19(
        gz_peak=6.0, gz_onset_rate=3.0, seat_tilt_deg=15.0, anti_g_strain=0.5,
        pilot_weight_kg=80.0, pilot_height_cm=175.0, pilot_age_y=35.0,
        baseline_vo2max=50.0, baseline_map_mmhg=90.0,
        fio2_inspired=0.21, sao2_baseline=0.97,
        sex_male=True, hypocapnia_flag=False,
        anti_g_suit_class=AntiGSuit.ATAGS,
        retinal_baseline_perfusion=0.85, cerebral_autoreg_gain=1.0,
        pulmonary_shunt_baseline=0.02, baseline_hb_g_dl=14.5,
        ambient_temp_c=22.0,
    )


def test_run_record_accepts_none_time_to_gloc_s() -> None:
    """Pulse-fidelity record: cerebral_o2_min populated, time_to_gloc_s None."""
    rec = RunRecord(
        run_id="pulse-1",
        fidelity=Fidelity.HIGH,
        features=_baseline_features(),
        time_to_gloc_s=None,
        cerebral_o2_min=0.95,
        engine_version="pulse-4.3.1",
    )
    assert rec.time_to_gloc_s is None
    assert rec.cerebral_o2_min == 0.95


def test_run_record_accepts_none_cerebral_o2_min() -> None:
    """CGEM-fidelity record: time_to_gloc_s populated, cerebral_o2_min None."""
    rec = RunRecord(
        run_id="cgem-1",
        fidelity=Fidelity.LOW,
        features=_baseline_features(),
        time_to_gloc_s=12.0,
        cerebral_o2_min=None,
        engine_version="cgem-surrogate-v1",
    )
    assert rec.time_to_gloc_s == 12.0
    assert rec.cerebral_o2_min is None


def test_parquet_roundtrip_with_none_fields() -> None:
    """Two records, one with each field None, must roundtrip via parquet."""
    feats = _baseline_features()
    cgem_rec = RunRecord(
        run_id="cgem-1", fidelity=Fidelity.LOW, features=feats,
        time_to_gloc_s=12.0, cerebral_o2_min=None,
        engine_version="cgem-surrogate-v1",
    )
    pulse_rec = RunRecord(
        run_id="pulse-1", fidelity=Fidelity.HIGH, features=feats,
        time_to_gloc_s=None, cerebral_o2_min=0.95,
        engine_version="pulse-4.3.1",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "records.parquet"
        write_records_parquet([cgem_rec, pulse_rec], path)
        df = read_records_parquet(path)
    assert len(df) == 2
    cgem_row = df[df["run_id"] == "cgem-1"].iloc[0]
    pulse_row = df[df["run_id"] == "pulse-1"].iloc[0]
    assert cgem_row["time_to_gloc_s"] == 12.0
    assert pd.isna(cgem_row["cerebral_o2_min"])
    assert pd.isna(pulse_row["time_to_gloc_s"])
    assert pulse_row["cerebral_o2_min"] == 0.95
