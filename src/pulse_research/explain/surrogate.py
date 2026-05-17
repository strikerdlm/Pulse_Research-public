"""XGBoost surrogate distilled from the MF-GP posterior mean.

The intended use is::

    mu_high, _ = mfgp.predict(X_design, fidelity="high")
    surr = fit_xgb_surrogate(X_design, mu_high, AXIS_NAMES)
    attr = shap_explain(surr, X_design)

so the SHAP explainer sees a fast tree model with the same input → output
shape as the GP, sidestepping the lack of a native TreeExplainer for GPs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from xgboost import XGBRegressor


@dataclass
class XGBSurrogate:
    """A fitted XGBoost surrogate + provenance for the SHAP step."""

    model: XGBRegressor
    feature_names: list[str]
    train_mae: float


def fit_xgb_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = 42,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> XGBSurrogate:
    """Fit an XGBoost regressor that distills a smooth target ``y``.

    The intended use is distilling an MF-GP posterior mean for SHAP
    attribution; the defaults are tuned for that workload, not for raw
    regression accuracy on noisy data:

    - ``n_estimators=400`` + ``learning_rate=0.05`` favors a larger ensemble
      of moderately-deep trees with small step size, which matches the smooth
      GP target without over-fitting to the design-point grid.
    - ``max_depth=6`` is shallow enough to keep TreeExplainer fast on
      ~20k-row Saltelli designs.
    - ``tree_method="hist"`` is XGBoost 2.x's modern default; CI-friendly
      (no GPU) and fast on 11-axis problems.
    - ``n_jobs=1`` keeps thread-count deterministic across CI machines.

    Override the kwargs when distilling a non-smooth target.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X / y row count mismatch: {X.shape[0]} vs {y.shape[0]}"
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names length {len(feature_names)} != X cols {X.shape[1]}"
        )

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        tree_method="hist",
        n_jobs=1,
    )
    model.fit(X, y)
    pred = model.predict(X)
    mae = float(np.mean(np.abs(pred - y)))
    return XGBSurrogate(
        model=model,
        feature_names=list(feature_names),
        train_mae=mae,
    )
