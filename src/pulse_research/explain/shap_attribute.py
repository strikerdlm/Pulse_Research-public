"""SHAP TreeExplainer over the XGBoost surrogate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap

from pulse_research.explain.surrogate import XGBSurrogate


@dataclass
class ShapAttribution:
    """Per-sample SHAP values + global mean(|SHAP|) ranking."""

    feature_names: list[str]
    values: np.ndarray
    base_value: float
    mean_abs: np.ndarray


def shap_explain(surrogate: XGBSurrogate, X: np.ndarray) -> ShapAttribution:
    """Compute SHAP values for ``X`` under ``surrogate.model``.

    TreeExplainer + XGBoost is deterministic for fixed inputs, so identical
    ``surrogate`` and ``X`` produce bit-identical SHAP arrays.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if X.shape[1] != len(surrogate.feature_names):
        raise ValueError(
            f"feature_names length {len(surrogate.feature_names)} != "
            f"X cols {X.shape[1]}"
        )

    explainer = shap.TreeExplainer(surrogate.model)
    expl = explainer(X)
    values = np.asarray(expl.values, dtype=float)
    base_values = np.asarray(expl.base_values, dtype=float)
    base = float(base_values.mean())
    mean_abs = np.mean(np.abs(values), axis=0)
    return ShapAttribution(
        feature_names=list(surrogate.feature_names),
        values=values,
        base_value=base,
        mean_abs=mean_abs,
    )
