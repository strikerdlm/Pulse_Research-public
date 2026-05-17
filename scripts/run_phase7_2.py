#!/usr/bin/env python3
"""Phase 7.2 orchestrator: twin single-fidelity GPs + joint sensitivity.

Loads ``data/run_records_phase7_1b.parquet`` and runs, for each arm
independently:

    GP fit -> split-conformal + CQR + Mondrian -> Sobol on posterior mean
    -> XGB distillation -> SHAP

Hard rule (spec §1): every research number lands in /root/repos/exports/
or docs/research/phase7_2_validation.md only after a live computation
against the real 7.1b parquet. Tests may use mock data; research outputs
may not.

Usage::

    python scripts/run_phase7_2.py \\
        --parquet data/run_records_phase7_1b.parquet \\
        --out-dir /root/repos/exports/ \\
        [--seed 42] [--alpha 0.10] [--sobol-n-base 8192]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulse_research.conformal import ConformalWrapper, calibrate
from pulse_research.explain.shap_attribute import ShapAttribution, shap_explain
from pulse_research.explain.surrogate import fit_xgb_surrogate
from pulse_research.orchestration.common import (
    AXIS_NAMES,
    FIO2_FEAT_IDX,
    ArmResult,
    fit_arm,
    load_paired_parquet,
)
from pulse_research.orchestration.common import (
    DATE_PREFIX as _DATE_PREFIX,
)
from pulse_research.physiology import (
    fio2_to_representative_altitude_m,
    g_loc_time,
    niermeyer_spo2,
)
from pulse_research.sensitivity.analyze import (
    SobolIndices,
    analyze_design,
    st_stability,
)
from pulse_research.sensitivity.sobol_design import build_design
from pulse_research.sensitivity.strata import fio2_tier
from pulse_research.surrogate import GPModel


def calibrate_arm(
    arm: ArmResult,
    *,
    alpha: float = 0.10,
    method: str = "cqr",
) -> tuple[ConformalWrapper, dict[str, float]]:
    """Calibrate conformal thresholds + compute empirical coverage on test fold.

    Mondrian strata = FiO2 tier per row. If the calibration fold of a stratum
    has < 30 rows, the wrapper falls back to marginal conformal for that
    stratum and a marginal threshold is used at predict time — recorded but
    not silently substituted.
    """
    X_cal_full = arm.X[arm.idx_calib]
    X_cal = arm.slice_active(X_cal_full)
    y_cal = arm.y[arm.idx_calib]
    # Strata are determined by the FiO2 column of the FULL design (regardless
    # of which subset the GP was fit on).
    strata_cal = fio2_tier(X_cal_full[:, FIO2_FEAT_IDX])
    # Marginal fallback computed first so it can be recorded in the stratified
    # wrapper for strata that did not appear in the calibration fold.
    marginal_wrapper = calibrate(
        arm.gp_model, X_cal, y_cal, alpha=alpha, method=method,  # type: ignore[arg-type]
    )
    wrapper = calibrate(
        arm.gp_model,
        X_cal,
        y_cal,
        alpha=alpha,
        method=method,  # type: ignore[arg-type]
        strata=strata_cal,
    )
    # Record (not silently substitute) marginal threshold for unseen strata.
    wrapper.strata_thresholds["_marginal"] = marginal_wrapper.strata_thresholds["_marginal"]
    wrapper.n_calib_per_stratum["_marginal"] = marginal_wrapper.n_calib_per_stratum["_marginal"]
    X_te_full = arm.X[arm.idx_test]
    X_te = arm.slice_active(X_te_full)
    y_te = arm.y[arm.idx_test]
    strata_te = fio2_tier(X_te_full[:, FIO2_FEAT_IDX])
    seen_real = set(wrapper.strata_thresholds) - {"_marginal"}
    strata_te_safe = np.where(np.isin(strata_te, list(seen_real)), strata_te, "_marginal")
    coverage = wrapper.coverage(X_te, y_te, strata=strata_te_safe)
    # Also compute marginal coverage explicitly for validation log.
    marginal_cov = marginal_wrapper.coverage(X_te, y_te)
    coverage = {**coverage, "_marginal": marginal_cov["_marginal"]}
    return wrapper, coverage


def analyze_arm(
    arm: ArmResult,
    *,
    sobol_n_base: int = 8192,
    seed: int = 42,
) -> tuple[SobolIndices, float, ShapAttribution, float]:
    """Sobol on posterior mean of a fresh Saltelli design + SHAP on XGB distillation.

    Sobol uses ``num_resamples=500`` (matches Phase 6 settings).
    SHAP fits an XGBRegressor on the GP posterior mean over the arm's
    full design (``arm.X``) and reports per-axis mean(|SHAP|).
    """
    # Sobol design is over the full 11 axes (the evaluation surface is the
    # 11-D Saltelli box). GP predictions slice the design to the active
    # subset; the resulting posterior is flat in inactive axes, which
    # correctly yields S1 / ST ~ 0 for those axes — the scientific finding
    # ("Pulse channel responds to fio2_inspired and sao2_baseline only").
    sobol_X, _ = build_design(n_base=sobol_n_base, seed=seed)
    mu_sobol, _ = arm.gp_model.predict(arm.slice_active(sobol_X))
    sobol = analyze_design(
        mu_sobol, num_resamples=500, seed=seed, calc_second_order=True,
    )
    stability = st_stability(sobol)

    # SHAP fits the XGB on the full 11-axis design with GP posterior mean as
    # target. The GP is fit on (possibly) a subset, so its predictions vary
    # only along the active axes; SHAP correctly attributes ~zero importance
    # to the rest.
    mu_design, _ = arm.gp_model.predict(arm.slice_active(arm.X))
    xgb = fit_xgb_surrogate(arm.X, mu_design, AXIS_NAMES, seed=seed)
    shap_attr = shap_explain(xgb, arm.X)
    return sobol, stability, shap_attr, xgb.train_mae


def compute_external_anchors(arm: ArmResult) -> dict[str, Any]:
    """Compare GP posterior predictions against published closed-form anchors.

    For CGEM arm (time_to_gloc_s target):
      - Stoll/Whinnery 2006 g_loc_time at each row's gz_peak (normobaric).
      - Per-row absolute residual: |GP_predicted_time - stoll_time|.
      - Excludes rows where stoll_time is inf (gz <= 3.0).
    For Pulse arm (cerebral_o2_min target):
      - Niermeyer 2019 spo2 at the row's representative altitude (per FiO2 tier).
      - Per-row absolute residual: |GP_predicted_O2 - niermeyer_spo2|.

    Result keys:
      - 'anchor_name', 'arm', 'n_compared', 'mae_vs_anchor', 'median_abs_diff'.

    Notes
    -----
    These anchors are NOT a primary validation target -- they're
    sanity-check comparisons against published equations. The GP
    posteriors model different physiology (CGEM models AGSM/tilt
    effects on time; Niermeyer applies to acclimatized humans).
    Wide deviations are not necessarily errors; they signal where
    the manuscript's Discussion must explain the model differences.
    """
    X = arm.X
    if arm.arm_name == "cgem":
        # Stoll curve at each Gz_peak (col 0 of the 11-axis design).
        gz_col = AXIS_NAMES.index("gz_peak")
        gz = X[:, gz_col]
        anchor_vals = np.array(
            [g_loc_time(float(g)) for g in gz], dtype=float
        )
        # Filter out infinite Stoll values (rows below threshold).
        finite_mask = np.isfinite(anchor_vals)
        if not np.any(finite_mask):
            return {
                "anchor_name": "stoll_whinnery_2006",
                "arm": arm.arm_name,
                "n_compared": 0,
                "mae_vs_anchor": float("nan"),
                "median_abs_diff": float("nan"),
            }
        mu_pred, _ = arm.gp_model.predict(arm.slice_active(X[finite_mask]))
        diff = np.abs(mu_pred - anchor_vals[finite_mask])
        return {
            "anchor_name": "stoll_whinnery_2006",
            "arm": arm.arm_name,
            "n_compared": int(finite_mask.sum()),
            "mae_vs_anchor": float(np.mean(diff)),
            "median_abs_diff": float(np.median(diff)),
        }
    # Pulse arm
    fio2_col = AXIS_NAMES.index("fio2_inspired")
    fio2 = X[:, fio2_col]
    anchor_vals = np.array(
        [
            niermeyer_spo2(fio2_to_representative_altitude_m(float(f)))
            for f in fio2
        ],
        dtype=float,
    )
    mu_pred, _ = arm.gp_model.predict(arm.slice_active(X))
    diff = np.abs(mu_pred - anchor_vals)
    return {
        "anchor_name": "niermeyer_tushaus_2019",
        "arm": arm.arm_name,
        "n_compared": len(diff),
        "mae_vs_anchor": float(np.mean(diff)),
        "median_abs_diff": float(np.median(diff)),
    }


def _sobol_to_dict(sobol: SobolIndices, stability: float) -> dict[str, Any]:
    return {
        "feature_names": sobol.names,
        "S1": sobol.S1.tolist(),
        "S1_conf": sobol.S1_conf.tolist(),
        "ST": sobol.ST.tolist(),
        "ST_conf": sobol.ST_conf.tolist(),
        "n_resamples": sobol.n_resamples,
        "stability": float(stability),
    }


def _shap_to_dict(shap_attr: ShapAttribution, xgb_train_mae: float) -> dict[str, Any]:
    return {
        "feature_names": shap_attr.feature_names,
        "mean_abs": shap_attr.mean_abs.tolist(),
        "base_value": float(shap_attr.base_value),
        "xgb_train_mae": float(xgb_train_mae),
    }


def _coverage_to_dict(
    conf: ConformalWrapper, coverage: dict[str, float]
) -> dict[str, Any]:
    return {
        "alpha": conf.alpha,
        "method": conf.method,
        "marginal": coverage.get("_marginal"),
        "per_stratum": {
            k: v for k, v in coverage.items() if k != "_marginal"
        },
        "n_calib_per_stratum": conf.n_calib_per_stratum,
        "strata_thresholds": conf.strata_thresholds,
    }


def _build_sobol_tornado_option(sobol: SobolIndices) -> dict[str, Any]:
    # ranked by ST descending
    order = list(np.argsort(-sobol.ST))
    return {
        "title": {"text": "Sobol — S1 vs ST (ranked by ST)"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["S1", "ST"]},
        "grid": {"left": 140, "right": 60, "top": 60, "bottom": 40},
        "xAxis": {"type": "value"},
        "yAxis": {
            "type": "category",
            "data": [sobol.names[i] for i in order],
        },
        "series": [
            {"name": "S1", "type": "bar",
             "data": [float(sobol.S1[i]) for i in order]},
            {"name": "ST", "type": "bar",
             "data": [float(sobol.ST[i]) for i in order]},
        ],
    }


def _build_shap_bar_option(shap_attr: ShapAttribution) -> dict[str, Any]:
    order = list(np.argsort(-shap_attr.mean_abs))
    return {
        "title": {"text": "mean(|SHAP|) — XGB distillation of GP posterior mean"},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 140, "right": 60, "top": 60, "bottom": 40},
        "xAxis": {"type": "value"},
        "yAxis": {
            "type": "category",
            "data": [shap_attr.feature_names[i] for i in order],
        },
        "series": [
            {"type": "bar",
             "itemStyle": {"color": "#7fb4c9"},
             "data": [float(shap_attr.mean_abs[i]) for i in order]},
        ],
    }


def _build_coverage_panel_option(
    alpha: float, coverage: dict[str, float]
) -> dict[str, Any]:
    items = [(k, v) for k, v in coverage.items()]
    return {
        "title": {
            "text": f"Empirical coverage (target {1.0 - alpha:.0%})"
        },
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 120, "right": 60, "top": 60, "bottom": 40},
        "xAxis": {"type": "value", "min": 0.0, "max": 1.0},
        "yAxis": {"type": "category", "data": [k for k, _ in items]},
        "series": [
            {"type": "bar",
             "data": [float(v) for _, v in items],
             "markLine": {
                 "data": [{"xAxis": 1.0 - alpha,
                           "label": {"formatter": "nominal"}}]
             }},
        ],
    }


def write_arm_artifacts(
    *,
    arm_name: str,
    out_dir: Path,
    date_prefix: str,
    conformal: ConformalWrapper,
    coverage: dict[str, float],
    sobol: SobolIndices,
    stability: float,
    shap_attr: ShapAttribution,
    xgb_train_mae: float,
    gp_model: GPModel,
) -> list[Path]:
    """Write 6 JSONs + 1 state pickle for this arm. Returns paths in stable order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{date_prefix}_phase7_2_{arm_name}"
    paths: list[Path] = []

    pairs: list[tuple[str, dict[str, Any]]] = [
        ("_sobol.json", _sobol_to_dict(sobol, stability)),
        ("_shap.json", _shap_to_dict(shap_attr, xgb_train_mae)),
        ("_coverage.json", _coverage_to_dict(conformal, coverage)),
        ("_sobol_tornado_option.json", _build_sobol_tornado_option(sobol)),
        ("_shap_bar_option.json", _build_shap_bar_option(shap_attr)),
        ("_coverage_panel_option.json", _build_coverage_panel_option(
            conformal.alpha, coverage,
        )),
    ]
    for suffix, payload in pairs:
        p = base.with_name(base.name + suffix)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
        paths.append(p)

    # State pickle — infrastructure for Phase 7.3 (spec §10).
    state_path = base.with_name(base.name + "_state.pkl")
    with state_path.open("wb") as f:
        pickle.dump({"gp": gp_model, "conformal": conformal}, f)
    paths.append(state_path)
    return paths


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/root/repos/exports/"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--sobol-n-base", type=int, default=8192)
    ap.add_argument("--no-save-state", action="store_true")
    return ap.parse_args(argv)


def write_validation_log(
    log_path: Path,
    *,
    parquet_path: Path,
    seed: int,
    alpha: float,
    sobol_n_base: int,
    results: dict[str, dict[str, Any]],
) -> None:
    """Emit docs/research/phase7_2_validation.md with per-arm numbers + audit lines.

    Every reported number is traced to its source (real parquet or live GP
    posterior) per spec §1.8.
    """
    lines: list[str] = []
    add = lines.append
    add("# Phase 7.2 Validation")
    add("")
    add(
        f"Twin single-fidelity GP analysis of "
        f"`{parquet_path}` (read 2026-05-15)."
    )
    add(f"Seed {seed}, alpha {alpha}, Sobol N_base {sobol_n_base}.")
    add("")
    add("## Per-arm results")
    add("")
    for arm_name, r in results.items():
        add(f"### {arm_name.upper()} arm")
        add("")
        active = r.get("active_axes")
        if active is None:
            add("- active axes: all 11 (full ARD)")
        else:
            add(f"- active axes: {active} ({len(active)}-D GP fit)")
        add(f"- train MAE: {r['train_mae']:.6f} (from real {parquet_path.name})")
        add(
            f"- marginal coverage: {r['coverage'].get('_marginal', float('nan')):.4f}"
            f" (target {1.0 - alpha:.0%})"
        )
        for k, v in r["coverage"].items():
            if k == "_marginal":
                continue
            add(f"- coverage[{k}]: {v:.4f} (real test fold)")
        add(f"- ST stability: {r['stability']:.4f} (real GP posterior mean)")
        add(f"- XGB distillation train MAE: {r['xgb_train_mae']:.6f}")
        add(
            "- top-5 Sobol ST: "
            + ", ".join(
                f"{name}={val:.4f}"
                for name, val in r["top_sobol_st"]
            )
        )
        add(
            "- top-5 mean(|SHAP|): "
            + ", ".join(
                f"{name}={val:.4f}"
                for name, val in r["top_shap"]
            )
        )
        add("")
    add("## External anchor comparison")
    add("")
    add(
        "Independent published-equation anchors (no fitted parameters; "
        "ported from strikerdlm/HumanPerformanceCalcs, verified "
        "2026-05-15 via the aerospace-calculators skill)."
    )
    add("")
    for arm_name, r in results.items():
        a = r["external_anchor"]
        add(
            f"- {arm_name.upper()} arm vs {a['anchor_name']}: "
            f"MAE = {a['mae_vs_anchor']:.4f} on {a['n_compared']} rows; "
            f"median |diff| = {a['median_abs_diff']:.4f}"
        )
    add("")
    add(
        "Anchors are sanity checks, not pass/fail targets -- the GP "
        "posteriors model superset physiology (AGSM, tilt, suit), so "
        "non-trivial deviations are expected and recorded for the "
        "manuscript's Discussion."
    )
    add("")
    add("## No-mock-data audit")
    add("")
    add(
        "Every number above is derived from a live computation on "
        f"`{parquet_path}` or from a GP posterior fit on it. No "
        "imputed values; no fallback constants. Hard rule spec §1."
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n")


def _top_k_pairs(
    names: list[str], values: np.ndarray, k: int = 5
) -> list[tuple[str, float]]:
    order = list(np.argsort(-values))[:k]
    return [(names[i], float(values[i])) for i in order]


def _all_results_written(out_dir: Path) -> bool:
    """True iff both arms' state pickles exist — the last file write_arm_artifacts() makes.

    Using existence of the terminal state pickle per arm avoids over-counting
    when the user runs with ``tee`` and the log file lands in the same directory.
    """
    cgem_state = out_dir / f"{_DATE_PREFIX}_phase7_2_cgem_state.pkl"
    pulse_state = out_dir / f"{_DATE_PREFIX}_phase7_2_pulse_state.pkl"
    return cgem_state.is_file() and pulse_state.is_file()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".phase7_2_running"
    sentinel.touch()
    try:
        cgem_df, pulse_df = load_paired_parquet(args.parquet)
        print(f"Loaded {len(cgem_df)} CGEM + {len(pulse_df)} Pulse rows")

        results: dict[str, dict[str, Any]] = {}
        # Per-arm active-axis subset. CGEM consumes all 11 axes via its
        # surrogate; Pulse's row_fn (pulse_glue.py) consumes only FiO2 and
        # SaO2 (the other 9 axes are dropped before invoking Pulse). Fitting
        # an 11-axis ARD GP on Pulse's 2-axis function destabilizes the
        # marginal log-likelihood — see spec §3.2 active-axes rationale.
        arm_configs: list[
            tuple[str, pd.DataFrame, str, tuple[str, ...] | None]
        ] = [
            ("cgem", cgem_df, "time_to_gloc_s", None),
            ("pulse", pulse_df, "cerebral_o2_min",
             ("fio2_inspired", "sao2_baseline")),
        ]
        for arm_name, df, target_col, active_axes in arm_configs:
            print(f"--- arm: {arm_name} ---")
            if active_axes is not None:
                print(f"  active axes: {active_axes}")
            arm = fit_arm(
                arm_name, df,
                target_col=target_col, seed=args.seed,
                active_axes=active_axes,
            )
            print(f"  train MAE: {arm.train_mae:.6f}")
            conf, coverage = calibrate_arm(arm, alpha=args.alpha)
            print(f"  marginal coverage: {coverage.get('_marginal', float('nan')):.4f}")
            sobol, stability, shap_attr, xgb_train_mae = analyze_arm(
                arm, sobol_n_base=args.sobol_n_base, seed=args.seed
            )
            print(f"  ST stability: {stability:.4f}")
            print(f"  XGB distill MAE: {xgb_train_mae:.6f}")
            anchors = compute_external_anchors(arm)
            print(
                f"  anchor {anchors['anchor_name']}: "
                f"MAE = {anchors['mae_vs_anchor']:.4f} "
                f"over {anchors['n_compared']} rows"
            )
            paths = write_arm_artifacts(
                arm_name=arm_name,
                out_dir=out_dir,
                date_prefix=_DATE_PREFIX,
                conformal=conf,
                coverage=coverage,
                sobol=sobol,
                stability=stability,
                shap_attr=shap_attr,
                xgb_train_mae=xgb_train_mae,
                gp_model=arm.gp_model,
            )
            print(f"  wrote {len(paths)} files")
            results[arm_name] = {
                "train_mae": arm.train_mae,
                "coverage": coverage,
                "stability": stability,
                "xgb_train_mae": xgb_train_mae,
                "top_sobol_st": _top_k_pairs(sobol.names, sobol.ST, k=5),
                "top_shap": _top_k_pairs(
                    shap_attr.feature_names, shap_attr.mean_abs, k=5
                ),
                "external_anchor": anchors,
                "active_axes": (
                    list(arm.active_axes) if arm.active_axes else None
                ),
            }

        log_path = Path(
            os.environ.get(
                "PHASE7_2_VALIDATION_LOG",
                "docs/research/phase7_2_validation.md",
            )
        )
        write_validation_log(
            log_path,
            parquet_path=args.parquet,
            seed=args.seed,
            alpha=args.alpha,
            sobol_n_base=args.sobol_n_base,
            results=results,
        )
        print(f"Wrote validation log to {log_path}")
    finally:
        if sentinel.exists() and _all_results_written(out_dir):
            sentinel.unlink()


if __name__ == "__main__":
    main()
