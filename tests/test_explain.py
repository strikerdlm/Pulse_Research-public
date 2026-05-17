"""Tests for the XGBoost surrogate + SHAP attribution layer.

CI installs the ``explain`` extra (xgboost + shap); local devs without it
will see these tests skip via importorskip."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")

from pulse_research.explain.shap_attribute import shap_explain
from pulse_research.explain.surrogate import fit_xgb_surrogate


def test_surrogate_fits_linear_target_within_0_05_mae() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(200, 3))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.normal(0.0, 0.01, size=200)
    surr = fit_xgb_surrogate(X, y, feature_names=["a", "b", "c"], seed=42)
    pred = surr.model.predict(X)
    assert float(np.mean(np.abs(pred - y))) < 0.05
    assert surr.feature_names == ["a", "b", "c"]
    assert surr.train_mae < 0.05


def test_surrogate_rejects_feature_name_count_mismatch() -> None:
    X = np.zeros((10, 3))
    y = np.zeros(10)
    with pytest.raises(ValueError, match="feature_names length"):
        fit_xgb_surrogate(X, y, feature_names=["a", "b"])


def test_surrogate_rejects_non_2d_X() -> None:
    X = np.zeros(10)
    y = np.zeros(10)
    with pytest.raises(ValueError, match="X must be 2-D"):
        fit_xgb_surrogate(X, y, feature_names=["a"])


def test_surrogate_rejects_row_count_mismatch() -> None:
    X = np.zeros((10, 2))
    y = np.zeros(9)
    with pytest.raises(ValueError, match="row count"):
        fit_xgb_surrogate(X, y, feature_names=["a", "b"])


def test_shap_ranks_dominant_feature_first() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(200, 4))
    # y depends almost entirely on column 0; columns 1-3 are noise.
    y = 5.0 * X[:, 0] + 0.001 * X[:, 1] + rng.normal(0.0, 0.01, size=200)
    surr = fit_xgb_surrogate(
        X, y, feature_names=["main", "n1", "n2", "n3"], seed=42,
    )
    attr = shap_explain(surr, X)
    assert attr.values.shape == (200, 4)
    assert attr.feature_names == ["main", "n1", "n2", "n3"]
    assert attr.mean_abs[0] > attr.mean_abs[1:].max()


def test_shap_is_deterministic_under_fixed_seed() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(50, 3))
    y = X.sum(axis=1)
    surr_a = fit_xgb_surrogate(X, y, feature_names=["a", "b", "c"], seed=42)
    surr_b = fit_xgb_surrogate(X, y, feature_names=["a", "b", "c"], seed=42)
    attr_a = shap_explain(surr_a, X)
    attr_b = shap_explain(surr_b, X)
    np.testing.assert_allclose(attr_a.values, attr_b.values)
    np.testing.assert_allclose(attr_a.mean_abs, attr_b.mean_abs)
    # also verify explainer determinism alone (same fitted surrogate, two calls):
    attr_a_again = shap_explain(surr_a, X)
    np.testing.assert_array_equal(attr_a.values, attr_a_again.values)


def test_shap_rejects_feature_count_mismatch() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(20, 3))
    y = X.sum(axis=1)
    surr = fit_xgb_surrogate(X, y, feature_names=["a", "b", "c"], seed=42)
    X_wrong = rng.uniform(0.0, 1.0, size=(5, 2))
    with pytest.raises(ValueError, match="feature_names length"):
        shap_explain(surr, X_wrong)


def test_shap_rejects_non_2d_X() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(20, 3))
    y = X.sum(axis=1)
    surr = fit_xgb_surrogate(X, y, feature_names=["a", "b", "c"], seed=42)
    X_1d = np.zeros(3)
    with pytest.raises(ValueError, match="X must be 2-D"):
        shap_explain(surr, X_1d)


def test_surrogate_rejects_non_1d_y() -> None:
    X = np.zeros((10, 2))
    y = np.zeros((10, 2))
    with pytest.raises(ValueError, match="y must be 1-D"):
        fit_xgb_surrogate(X, y, feature_names=["a", "b"])
