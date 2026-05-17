"""End-to-end tests for the Phase 7.3 orchestrator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_mock_parquet(tmp_path: Path, n_per_arm: int = 192) -> Path:
    """Write a mock 7.1b-shaped parquet for tests only (spec §1.2)."""
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
    t_cgem = np.clip(8.0 * (1.0 - (X[:, 0] - 1.0) / 8.0), 0.0, 8.7)
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
    import scripts.run_phase7_3 as m

    assert hasattr(m, "main")


def test_refit_arms_produces_two_armresults(tmp_path: Path) -> None:
    from scripts.run_phase7_3 import refit_arms

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    cgem_arm, pulse_arm = refit_arms(pq, seed=42)
    assert cgem_arm.gp_model.feature_dim == 11
    assert pulse_arm.gp_model.feature_dim == 2
    assert pulse_arm.active_axes == ("fio2_inspired", "sao2_baseline")


def test_mc_coverage_returns_marginal_and_per_stratum(tmp_path: Path) -> None:
    from scripts.run_phase7_3 import mc_coverage, refit_arms

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    cgem_arm, pulse_arm = refit_arms(pq, seed=42)
    result = mc_coverage(
        cgem_arm, pulse_arm, alpha=0.10, n_mc=256, seed=42,
    )
    for key in (
        "alpha", "n_mc", "marginal", "per_stratum",
        "n_test_per_stratum", "propagation_method",
    ):
        assert key in result, f"missing key {key} in {result!r}"
    assert 0.0 <= result["marginal"] <= 1.0
    assert isinstance(result["per_stratum"], dict)
    assert result["propagation_method"] == "monte_carlo_independent_draws"


def test_interaction_sobol_returns_full_matrix(tmp_path: Path) -> None:
    from scripts.run_phase7_3 import interaction_sobol, refit_arms

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    cgem_arm, pulse_arm = refit_arms(pq, seed=42)
    result = interaction_sobol(
        cgem_arm, pulse_arm, n_base=128, seed=42,
    )
    assert "S1" in result
    assert "ST" in result
    assert "S2" in result
    assert "stability" in result
    assert "headline_s2_gz_fio2" in result
    assert len(result["S1"]) == 11
    assert len(result["ST"]) == 11
    # S2 is 11x11
    assert len(result["S2"]) == 11
    assert all(len(row) == 11 for row in result["S2"])


def test_whinnery_anchor_returns_rank_metrics(tmp_path: Path) -> None:
    from scripts.run_phase7_3 import refit_arms, whinnery_anchor

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    cgem_arm, pulse_arm = refit_arms(pq, seed=42)
    result = whinnery_anchor(cgem_arm, pulse_arm)
    for key in (
        "n_compared", "mae", "spearman_rho", "gz_band",
        "onset_band", "fio2_band",
    ):
        assert key in result
    assert result["n_compared"] >= 0
    # spearman_rho is NaN when wf_time is constant (all rapid-regime rows in
    # mock data) or when n_compared < 2; otherwise must be in [-1, 1].
    rho = result["spearman_rho"]
    assert np.isnan(rho) or (-1.0 <= rho <= 1.0)


def test_write_phase7_3_artifacts_creates_ten_files(tmp_path: Path) -> None:
    from scripts.run_phase7_3 import (
        interaction_sobol,
        interaction_sobol_ablated,
        mc_coverage,
        refit_arms,
        split_conformal_coverage,
        whinnery_anchor,
        whinnery_bins_anchor,
        write_phase7_3_artifacts,
    )

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cgem_arm, pulse_arm = refit_arms(pq, seed=42)
    mc = mc_coverage(cgem_arm, pulse_arm, alpha=0.10, n_mc=128, seed=42)
    sobol = interaction_sobol(cgem_arm, pulse_arm, n_base=128, seed=42)
    wf = whinnery_anchor(cgem_arm, pulse_arm)
    wf_bins = whinnery_bins_anchor(cgem_arm, pulse_arm)
    ablation = interaction_sobol_ablated(
        cgem_arm, pulse_arm, n_base=128, seed=42,
    )
    conformal = split_conformal_coverage(cgem_arm, pulse_arm, alpha=0.10)
    paths = write_phase7_3_artifacts(
        cgem_arm=cgem_arm, pulse_arm=pulse_arm,
        out_dir=out_dir, date_prefix="2026-05-15",
        mc=mc, sobol=sobol, wf=wf,
        wf_bins=wf_bins, ablation=ablation, conformal=conformal,
    )
    expected_suffixes = [
        "_corrected_time_sobol.json",
        "_corrected_time_sobol_tornado_option.json",
        "_interaction_heatmap_option.json",
        "_mc_coverage.json",
        "_mc_coverage_panel_option.json",
        "_whinnery_anchor.json",
        "_corrected_time_distribution_option.json",
        "_whinnery_bins_anchor.json",
        "_coupling_ablation_sobol.json",
        "_split_conformal_coverage.json",
    ]
    for suffix in expected_suffixes:
        assert any(p.name.endswith(suffix) for p in paths), (
            f"missing artifact ending in {suffix} among {[p.name for p in paths]}"
        )
        for p in paths:
            if p.name.endswith(suffix):
                assert p.is_file()


def test_main_end_to_end_writes_seven_files_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_phase7_3 as orch

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    out_dir = tmp_path / "out"
    val_log = tmp_path / "validation.md"
    monkeypatch.setenv("PHASE7_3_VALIDATION_LOG", str(val_log))
    orch.main([
        "--parquet", str(pq),
        "--out-dir", str(out_dir),
        "--seed", "42",
        "--alpha", "0.10",
        "--n-mc", "128",
        "--sobol-n-base", "128",
    ])
    files = sorted(out_dir.iterdir())
    artifact_files = [p for p in files if not p.name.startswith(".")]
    assert len(artifact_files) == 10, (
        f"expected 10 artifacts, got {len(artifact_files)}: "
        f"{[f.name for f in artifact_files]}"
    )
    assert val_log.exists()
    sentinel = out_dir / ".phase7_3_running"
    assert not sentinel.exists(), "sentinel should be cleared on success"


def test_main_keeps_sentinel_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_phase7_3 as orch

    pq = _write_mock_parquet(tmp_path, n_per_arm=192)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def bomb(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic mid-pipeline failure")

    monkeypatch.setattr(orch, "refit_arms", bomb)
    with pytest.raises(RuntimeError):
        orch.main([
            "--parquet", str(pq),
            "--out-dir", str(out_dir),
            "--n-mc", "32",
            "--sobol-n-base", "64",
        ])
    sentinel = out_dir / ".phase7_3_running"
    assert sentinel.exists(), "sentinel must persist on crash"
