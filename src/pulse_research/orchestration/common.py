"""Shared orchestration helpers — arms, splits, parquet loading.

Extracted from ``scripts/run_phase7_2.py`` (2026-05-15) so Phase 7.3's
orchestrator imports the same code path. Zero functional change from
Phase 7.2; the move is the entire delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pulse_research.surrogate import GPModel, train_gp

DATE_PREFIX = "2026-05-15"

AXIS_NAMES: list[str] = [
    "gz_peak", "gz_onset_rate", "seat_tilt_deg", "anti_g_strain",
    "pilot_weight_kg", "pilot_height_cm", "pilot_age_y",
    "baseline_vo2max", "baseline_map_mmhg", "fio2_inspired",
    "sao2_baseline",
]
FEAT_COLUMNS: list[str] = [f"feat_{a}" for a in AXIS_NAMES]
FIO2_FEAT_IDX = AXIS_NAMES.index("fio2_inspired")


def load_paired_parquet(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the 7.1b parquet and split by fidelity.

    Returns the (CGEM, Pulse) sub-DataFrames sorted by run_id for determinism.
    """
    df = pd.read_parquet(path)
    if "fidelity" not in df.columns:
        raise ValueError(f"{path}: missing 'fidelity' column")
    cgem_df = df[df["fidelity"] == "low"].sort_values("run_id").reset_index(drop=True)
    pulse_df = df[df["fidelity"] == "high"].sort_values("run_id").reset_index(drop=True)
    if cgem_df.empty or pulse_df.empty:
        raise ValueError(
            f"{path}: empty arm — cgem rows={len(cgem_df)}, pulse rows={len(pulse_df)}"
        )
    return cgem_df, pulse_df


@dataclass
class ArmResult:
    arm_name: str
    gp_model: GPModel
    train_mae: float
    idx_train: np.ndarray
    idx_calib: np.ndarray
    idx_test: np.ndarray
    X: np.ndarray  # full design matrix (n, 11)
    y: np.ndarray  # full target vector (n,)
    active_axes: tuple[str, ...] | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def active_indices(self) -> list[int]:
        """Column indices of ``X`` actually consumed by the GP."""
        if self.active_axes is None:
            return list(range(self.X.shape[1]))
        return [AXIS_NAMES.index(a) for a in self.active_axes]

    def slice_active(self, X: np.ndarray) -> np.ndarray:
        """Return ``X`` projected onto the axes the GP was fit on."""
        if self.active_axes is None:
            return X
        return X[:, self.active_indices]


def random_split(
    n: int,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random 70/15/15 train/calibrate/test index split."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split ratios must sum to 1; got {ratios}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = round(n * ratios[0])
    n_cal = round(n * ratios[1])
    idx_train = perm[:n_train]
    idx_calib = perm[n_train : n_train + n_cal]
    idx_test = perm[n_train + n_cal :]
    return idx_train, idx_calib, idx_test


def fit_arm(
    arm_name: str,
    df: pd.DataFrame,
    *,
    target_col: str,
    seed: int = 42,
    active_axes: tuple[str, ...] | None = None,
) -> ArmResult:
    """Random-split the arm's rows, fit a GP on the train fold, report train MAE.

    Parameters
    ----------
    active_axes:
        Axis names actually consumed by the arm's oracle. ``None`` fits an
        11-axis ARD GP. A tuple subset (e.g. ``("fio2_inspired",
        "sao2_baseline")`` for the Pulse arm) fits the GP only on those
        columns — the structurally correct model when the oracle ignores
        other axes.
    """
    X = df[FEAT_COLUMNS].to_numpy()
    y = df[target_col].to_numpy()
    if np.any(np.isnan(y)):
        raise ValueError(
            f"arm {arm_name}: target column {target_col} contains NaN — orthogonal-oracle violation"
        )
    if active_axes is not None:
        missing = [a for a in active_axes if a not in AXIS_NAMES]
        if missing:
            raise ValueError(
                f"arm {arm_name}: active_axes contains unknown names {missing!r}"
            )
    idx_train, idx_calib, idx_test = random_split(len(y), seed=seed)
    active_idx = (
        [AXIS_NAMES.index(a) for a in active_axes]
        if active_axes is not None
        else None
    )
    X_train_gp = X[idx_train] if active_idx is None else X[idx_train][:, active_idx]
    gp = train_gp(X_train_gp, y[idx_train], seed=seed)
    train_mae = gp.residual_mae(X_train_gp, y[idx_train])
    return ArmResult(
        arm_name=arm_name,
        gp_model=gp,
        train_mae=train_mae,
        idx_train=idx_train,
        idx_calib=idx_calib,
        idx_test=idx_test,
        X=X,
        y=y,
        active_axes=active_axes,
    )
