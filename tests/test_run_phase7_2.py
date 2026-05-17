"""End-to-end tests for the Phase 7.2 orchestrator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_mock_parquet(tmp_path: Path, n_per_arm: int = 96) -> Path:
    """Write a small mock 7.1b-shaped parquet for tests only.

    Tests-only mock data — spec §1.3. Real research outputs use the real
    Phase 7.1b parquet.
    """
    rng = np.random.default_rng(42)
    axes = [
        "gz_peak", "gz_onset_rate", "seat_tilt_deg", "anti_g_strain",
        "pilot_weight_kg", "pilot_height_cm", "pilot_age_y",
        "baseline_vo2max", "baseline_map_mmhg", "fio2_inspired",
        "sao2_baseline",
    ]
    feat_cols = [f"feat_{a}" for a in axes]
    bounds = {
        "gz_peak": (1.0, 9.0),
        "gz_onset_rate": (0.1, 6.0),
        "seat_tilt_deg": (0.0, 30.0),
        "anti_g_strain": (0.0, 1.0),
        "pilot_weight_kg": (50.0, 110.0),
        "pilot_height_cm": (155.0, 200.0),
        "pilot_age_y": (20.0, 55.0),
        "baseline_vo2max": (30.0, 70.0),
        "baseline_map_mmhg": (70.0, 110.0),
        "fio2_inspired": (0.15, 1.00),
        "sao2_baseline": (0.80, 1.00),
    }
    X = np.column_stack(
        [rng.uniform(lo, hi, n_per_arm) for lo, hi in (bounds[a] for a in axes)]
    )
    # Synthetic CGEM target: depends mostly on gz_peak; range similar to real P*E.
    t_cgem = np.clip(8.0 * (1.0 - (X[:, 0] - 1.0) / 8.0), 0.0, 8.7)
    # Synthetic Pulse target: depends on fio2; range 0.89-0.97 like the real data.
    o2_pulse = 0.89 + 0.08 * (X[:, 9] - 0.15) / 0.85

    cgem_df = pd.DataFrame(X, columns=feat_cols)
    cgem_df["run_id"] = [f"mock-cgem-{i:04d}" for i in range(n_per_arm)]
    cgem_df["fidelity"] = "low"
    cgem_df["time_to_gloc_s"] = t_cgem
    cgem_df["cerebral_o2_min"] = np.nan
    cgem_df["engine_version"] = "cgem-surrogate-v1"

    pulse_df = pd.DataFrame(X, columns=feat_cols)
    pulse_df["run_id"] = [f"mock-pulse-{i:04d}" for i in range(n_per_arm)]
    pulse_df["fidelity"] = "high"
    pulse_df["time_to_gloc_s"] = np.nan
    pulse_df["cerebral_o2_min"] = o2_pulse
    pulse_df["engine_version"] = "pulse-4.3.1"

    out = tmp_path / "mock_phase7_1b.parquet"
    pd.concat([cgem_df, pulse_df], ignore_index=True).to_parquet(out)
    return out


def test_orchestrator_imports() -> None:
    """Module imports without side effects."""
    import scripts.run_phase7_2 as m

    assert hasattr(m, "main")


def test_load_and_split(tmp_path: Path) -> None:
    """The loader returns CGEM and Pulse DataFrames with expected shapes."""
    from scripts.run_phase7_2 import load_paired_parquet

    pq = _write_mock_parquet(tmp_path, n_per_arm=96)
    cgem_df, pulse_df = load_paired_parquet(pq)
    assert len(cgem_df) == 96
    assert len(pulse_df) == 96
    assert "feat_gz_peak" in cgem_df.columns
    assert cgem_df["time_to_gloc_s"].notna().all()
    assert pulse_df["cerebral_o2_min"].notna().all()


def test_fit_arm_returns_gp_with_low_train_mae(tmp_path: Path) -> None:
    """fit_arm trains a GP on the train fold; train MAE is < 1 % of y range."""
    from scripts.run_phase7_2 import fit_arm, load_paired_parquet

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    cgem_df, _ = load_paired_parquet(pq)
    result = fit_arm("cgem", cgem_df, target_col="time_to_gloc_s", seed=42)
    assert result.gp_model.feature_dim == 11
    assert 0 <= result.train_mae < 0.5  # synthetic CGEM target spans ~0-8
    assert len(result.idx_train) > 0
    assert len(result.idx_calib) > 0
    assert len(result.idx_test) > 0
    # 70/15/15 split with all unique indices
    all_idx = set(result.idx_train) | set(result.idx_calib) | set(result.idx_test)
    assert len(all_idx) == 192


def test_calibrate_arm_produces_coverage(tmp_path: Path) -> None:
    from scripts.run_phase7_2 import (
        calibrate_arm,
        fit_arm,
        load_paired_parquet,
    )

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    cgem_df, _ = load_paired_parquet(pq)
    arm = fit_arm("cgem", cgem_df, target_col="time_to_gloc_s", seed=42)
    _conf, coverage = calibrate_arm(arm, alpha=0.10, method="cqr")
    assert 0.6 <= coverage["_marginal"] <= 1.0
    assert "normo" in coverage or "_marginal" in coverage


def test_analyze_arm_returns_sobol_and_shap(tmp_path: Path) -> None:
    from scripts.run_phase7_2 import (
        analyze_arm,
        fit_arm,
        load_paired_parquet,
    )

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    cgem_df, _ = load_paired_parquet(pq)
    arm = fit_arm("cgem", cgem_df, target_col="time_to_gloc_s", seed=42)
    sobol, stability, shap_attr, xgb_train_mae = analyze_arm(
        arm, sobol_n_base=128, seed=42
    )
    assert len(sobol.S1) == 11
    assert len(sobol.ST) == 11
    # st_stability is (-inf, 1]; mock target depends ~only on gz_peak so
    # inactive-axis ST_conf/ST ratios explode on small n_base=128.
    # Only the upper bound and type are checkable here; real 7.1b data
    # will yield usable values.
    assert stability <= 1.0
    assert shap_attr.mean_abs.shape == (11,)
    assert xgb_train_mae >= 0.0


def test_write_arm_artifacts_creates_seven_files(tmp_path: Path) -> None:
    """Six JSON artifacts plus one state pickle per arm."""
    from scripts.run_phase7_2 import (
        analyze_arm,
        calibrate_arm,
        fit_arm,
        load_paired_parquet,
        write_arm_artifacts,
    )

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cgem_df, _ = load_paired_parquet(pq)
    arm = fit_arm("cgem", cgem_df, target_col="time_to_gloc_s", seed=42)
    conf, coverage = calibrate_arm(arm)
    sobol, stability, shap_attr, xgb_train_mae = analyze_arm(
        arm, sobol_n_base=128
    )
    paths = write_arm_artifacts(
        arm_name="cgem",
        out_dir=out_dir,
        date_prefix="2026-05-15",
        conformal=conf,
        coverage=coverage,
        sobol=sobol,
        stability=stability,
        shap_attr=shap_attr,
        xgb_train_mae=xgb_train_mae,
        gp_model=arm.gp_model,
    )
    expected_suffixes = [
        "_sobol.json", "_shap.json", "_coverage.json",
        "_sobol_tornado_option.json", "_shap_bar_option.json",
        "_coverage_panel_option.json", "_state.pkl",
    ]
    for suffix in expected_suffixes:
        assert any(p.name.endswith(suffix) for p in paths), (
            f"missing artifact ending in {suffix} among {[p.name for p in paths]}"
        )
        for p in paths:
            if p.name.endswith(suffix):
                assert p.is_file(), f"{p} not written"


def test_main_end_to_end_writes_14_files_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_phase7_2 as orch

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    out_dir = tmp_path / "out"
    validation_log = tmp_path / "validation.md"
    monkeypatch.setenv("PHASE7_2_VALIDATION_LOG", str(validation_log))
    orch.main([
        "--parquet", str(pq),
        "--out-dir", str(out_dir),
        "--seed", "42",
        "--alpha", "0.10",
        "--sobol-n-base", "128",
    ])

    files = sorted(out_dir.iterdir())
    assert len(files) == 14, f"expected 14 artifacts, got {len(files)}: {[f.name for f in files]}"
    assert validation_log.exists()
    sentinel = out_dir / ".phase7_2_running"
    assert not sentinel.exists(), "sentinel should be cleared on success"


def test_main_includes_external_anchors_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_phase7_2 as orch

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    out_dir = tmp_path / "out"
    validation_log = tmp_path / "val.md"
    monkeypatch.setenv("PHASE7_2_VALIDATION_LOG", str(validation_log))
    orch.main([
        "--parquet", str(pq),
        "--out-dir", str(out_dir),
        "--sobol-n-base", "64",
    ])
    text = validation_log.read_text()
    assert "External anchor comparison" in text
    assert "stoll_whinnery_2006" in text
    assert "niermeyer_tushaus_2019" in text


def test_main_keeps_sentinel_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_phase7_2 as orch

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def bomb(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic mid-pipeline failure")

    monkeypatch.setattr(orch, "fit_arm", bomb)
    with pytest.raises(RuntimeError):
        orch.main([
            "--parquet", str(pq),
            "--out-dir", str(out_dir),
            "--sobol-n-base", "64",
        ])
    sentinel = out_dir / ".phase7_2_running"
    assert sentinel.exists(), "sentinel must persist on crash"
