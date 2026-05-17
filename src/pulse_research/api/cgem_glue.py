"""Glue between the 11-axis Sobol design and the upstream CGEM Fortran wrapper.

Two responsibilities, kept in one small file:

1. :func:`design_row_to_centrifuge_params` is a pure mapping from a Sobol row
   to ``cgem_wrapper.run_cgem_centrifuge`` keyword arguments. The mapping is
   exact and tested against a stub ``PilotConfig`` class — no real CGEM
   needed.

2. :func:`make_cgem_row_fn` resolves the upstream wrapper module (either via
   the ``CGEM_ROOT`` env var with a sys.path injection, or via an injected
   module for tests) and returns the per-row callable consumed by
   :class:`pulse_research.api.runners.CGEMRunner`.

Why the env-var-driven sys.path injection: ``cgem_wrapper.py`` lives at the
CGEM repo root but is NOT part of the installed ``cgem-ext`` pip package
(see the upstream ``pyproject.toml`` ``[tool.setuptools.packages.find]``
include glob). A clean ``import cgem_wrapper`` requires the repo root on
``sys.path``. This is documented at the call site so an operator can debug
without spelunking.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pandas as pd

from pulse_research.api.runners import RowFn, RowOutput

CGEM_ENV_VAR = "CGEM_ROOT"


def design_row_to_centrifuge_params(
    row: np.ndarray,
    *,
    PilotConfigCls: type,
    hold_seconds: float = 15.0,
    g0: float = 1.0,
) -> dict[str, Any]:
    """Map an 11-axis Sobol row to ``run_cgem_centrifuge`` kwargs.

    Mapping (locked in
    ``docs/superpowers/plans/2026-05-14-phase-4-5-cgem-runner.md``):

    * row[0] gz_peak           → ``gmax``
    * row[1] gz_onset_rate     → ``rampup = max(0.1, (gmax - g0) / onset_rate)``
    * row[2] seat_tilt_deg     → ``PilotConfig.seat_tilt_deg``
    * row[3] anti_g_strain     → ``PilotConfig.agsm_effectiveness`` (clip 0..1)
    * row[4] pilot_weight_kg   → UNUSED (CGEM has no weight axis)
    * row[5] pilot_height_cm   → ``PilotConfig.height_cm``  (who_profile=None)
    * row[6] pilot_age_y       → UNUSED
    * row[7] baseline_vo2max   → UNUSED
    * row[8] baseline_map_mmhg → derive baseline systolic/diastolic
    * row[9] fio2_inspired     → IGNORED (CGEM has no FiO2 axis — the gap)
    * row[10] sao2_baseline    → IGNORED — same reason

    Defaults injected by this mapping: ``g0=1.0``, ``gmaxtime=hold_seconds``,
    ``rampdown=rampup``.
    """
    if row.ndim != 1 or row.shape[0] != 11:
        raise ValueError(f"row must be shape (11,); got {row.shape}")

    gz_peak = float(row[0])
    gz_onset_rate = float(row[1])
    seat_tilt = float(row[2])
    agsm = max(0.0, min(1.0, float(row[3])))
    height_cm = float(row[5])
    map_mmhg = float(row[8])

    rampup = max(0.1, (gz_peak - g0) / gz_onset_rate)

    config = PilotConfigCls(
        who_profile=None,
        height_cm=height_cm,
        seat_tilt_deg=seat_tilt,
        agsm_effectiveness=agsm,
        baseline_systolic_bp=map_mmhg + 20.0,
        baseline_diastolic_bp=max(40.0, map_mmhg - 10.0),
    )

    return {
        "g0": g0,
        "gmax": gz_peak,
        "gmaxtime": hold_seconds,
        "rampup": rampup,
        "rampdown": rampup,
        "config": config,
    }


def _resolve_cgem_module(cgem_root: str | None = None) -> ModuleType:
    """Inject CGEM repo root onto ``sys.path`` and import ``cgem_wrapper``."""
    if cgem_root is None:
        cgem_root = os.environ.get(CGEM_ENV_VAR)
    if cgem_root is None:
        raise RuntimeError(
            f"CGEM wrapper not configured. Set the {CGEM_ENV_VAR} env var to the "
            f"absolute path of the CAMI-Gz-Effects-Model-CGEM- repo root, or pass "
            f"a fake cgem_module to make_cgem_row_fn()."
        )
    root = str(Path(cgem_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import cgem_wrapper as cgem_module
    except ImportError as e:
        raise RuntimeError(
            f"Resolved {CGEM_ENV_VAR}={root} but could not import cgem_wrapper: {e}"
        ) from e
    return cast(ModuleType, cgem_module)


def make_cgem_row_fn(cgem_module: ModuleType | None = None) -> RowFn:
    """Build the per-row callable bound to ``run_cgem_centrifuge``.

    Parameters
    ----------
    cgem_module:
        Pre-resolved module exposing ``PilotConfig`` and
        ``run_cgem_centrifuge``. When ``None``, the module is resolved via
        ``CGEM_ROOT`` (see :func:`_resolve_cgem_module`).
    """
    module = cgem_module if cgem_module is not None else _resolve_cgem_module()
    pilot_cls = module.PilotConfig
    run_centrifuge = module.run_cgem_centrifuge

    def _row_fn(row: np.ndarray) -> RowOutput:
        try:
            kwargs = design_row_to_centrifuge_params(row, PilotConfigCls=pilot_cls)
            result, _tmp = run_centrifuge(**kwargs)
        except Exception as e:
            return RowOutput(time_to_gloc_s=None, cerebral_o2_min=None, error=str(e))
        t_gloc = getattr(result, "time_to_gloc_s", None)
        return RowOutput(
            time_to_gloc_s=float(t_gloc) if t_gloc is not None else None,
            cerebral_o2_min=None,
        )

    return _row_fn


def design_row_to_surrogate_features(row: np.ndarray) -> pd.DataFrame:
    """Project the 11-axis Pulse_Research design row onto the 17-d CGEM
    surrogate feature schema (``cgem_ext.ood.features.FEATURE_COLUMNS``).

    The 5 continuous pilot axes (weight, height, age, vo2max, MAP) and
    seat_tilt_deg project onto CGEM's scalar ``g_tolerance_multiplier``
    via a closed-form physiological formula (see spec Phase 7.1a).
    The clip range ``[0.85, 1.15]`` matches the three discrete levels
    present in CGEM's training data; values outside this band would
    force the Stage-2 conditional-time regressor to extrapolate. See
    Phase 7.1b spec §"Design choices" for the locked decision.
    Hypoxia axes (fio2_inspired, sao2_baseline) are dropped — CGEM is
    hypoxia-blind by design (orthogonal-oracle).
    """
    gz_peak, gz_rate, tilt, agsm, _w, _h, age, vo2, mapmm, _fio2, _sao2 = row

    g_tolerance_multiplier = float(np.clip(
        1.0
        + 0.005 * (mapmm - 90.0)
        + 0.010 * (vo2 - 50.0)
        - 0.005 * (age - 30.0)
        + 0.005 * tilt,
        a_min=0.85, a_max=1.15,
    ))

    if agsm < 0.1:
        cm_ordinal = 0.0
    elif agsm < 0.7:
        cm_ordinal = 1.0
    else:
        cm_ordinal = 2.0

    return pd.DataFrame([{
        "g_peak_abs": float(gz_peak),
        "dgdt_max_g_per_s": float(gz_rate),
        "profile_duration_s": 30.0,
        "dehydration_level": 0.0,
        "g_tolerance_multiplier": g_tolerance_multiplier,
        "gsuit_max_psi": 0.0,
        "gsuit_coverage_fraction": 0.0,
        "agsm_effectiveness": float(agsm),
        "pbg_max_mmhg": 0.0,
        "who_1": 0.0, "who_2": 0.0, "who_3": 0.0,
        "who_4": 0.0, "who_5": 0.0, "who_6": 0.0,
        "who_custom": 1.0,
        "cm_ordinal": cm_ordinal,
    }])


def make_cgem_surrogate_row_fn(
    parquet_path: Path,
    *,
    target: str = "time_to_gloc_s",
) -> RowFn:
    """Train a CGEM XGBoost surrogate from ``parquet_path`` and return a
    RowFn that translates Pulse_Research 11-axis design rows into CGEM's
    17-d feature schema and predicts the **conditional event-time** —
    ``E[time | event=1]`` — using the regressor-only channel.

    Loads the parquet once, drops error rows (``status != "ok"``), and trains a
    :class:`cgem_ext.surrogate.xgb.TwoStageXGBSurrogate`. The surrogate handles
    right-censoring internally (Stage 1 classifier trains on all rows; Stage 2
    regressor trains only on event-positive rows). Returns a closure that is
    cheap to call (~ms per row).

    The row_fn returns ``E[time | event=1]`` via ``predict_array`` — the
    Stage-2 conditional-time regressor's output. The narrow training-set
    range ``[5.4, 16.9]`` s is the correct empirical envelope for
    centrifuge G-LOC times (LOCINDTI per Whinnery & Forster 2013, with
    the 5 s minimum LOC-induction time observed across all 888 episodes
    in the WF2013 repository).

    **Channel choice rationale (updated 2026-05-16):** the original Phase
    7.1a triage adopted the marginal ``P(event) * E[time | event=1]``
    channel because the regressor-only range looked too narrow. Phase 7.3
    surfaced that this choice anti-correlates with WF2013 LOCINDTI in
    the rapid-onset regime (rho_ROR = -0.67): the P*E channel is
    monotonically increasing in Gz (P(event) saturates -> 1; the
    conditional-time floor at ~5 s dominates), while LOCINDTI is
    monotonically decreasing in Gz. The conditional-time channel is the
    physiologically correct quantity for direct LOCINDTI comparability.

    Per the orthogonal-oracle design (Phase 4.7), CGEM populates
    ``time_to_gloc_s`` only; the returned RowOutput has
    ``cerebral_o2_min=None``.
    """
    from cgem_ext.surrogate.xgb import build_surrogate

    df = pd.read_parquet(parquet_path)
    df = df[df["status"] == "ok"].copy()

    surrogate = build_surrogate(target).fit(df)

    def _row_fn(row: np.ndarray) -> RowOutput:
        features = design_row_to_surrogate_features(row)
        pred = surrogate.predict_array(
            features.to_numpy(dtype=float)
        )
        value = float(pred[0])
        return RowOutput(
            time_to_gloc_s=value,
            cerebral_o2_min=None,
            error=None,
        )

    return _row_fn
