"""Tests for the split / CQR / Mondrian conformal wrapper."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pulse_research.conformal import ConformalWrapper, calibrate
from pulse_research.surrogate.mfgp import MFGPModel, train_mfgp

from .synthetic import forrester_high, make_forrester_pair


@pytest.fixture(autouse=True)
def _deterministic_torch() -> None:
    torch.manual_seed(42)


def _fit_forrester(
    seed: int = 42,
) -> tuple[MFGPModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train MF-GP, return calibration and test cohorts on Forrester high-fid."""
    low_X, low_y, high_X_train, high_y_train = make_forrester_pair(
        n_low=80, n_high=12, seed=seed
    )
    model = train_mfgp(low_X, low_y, high_X_train, high_y_train, seed=seed)

    rng = np.random.default_rng(seed + 1)
    X_cal = rng.uniform(0.0, 1.0, size=(120, 1))
    y_cal = forrester_high(X_cal.ravel())
    X_test = rng.uniform(0.0, 1.0, size=(400, 1))
    y_test = forrester_high(X_test.ravel())
    return model, X_cal, y_cal, X_test, y_test


def test_calibrate_returns_wrapper() -> None:
    model, X_cal, y_cal, _, _ = _fit_forrester()
    w = calibrate(model, X_cal, y_cal, alpha=0.10, method="split")
    assert isinstance(w, ConformalWrapper)
    assert w.method == "split"
    assert w.alpha == 0.10
    assert "_marginal" in w.strata_thresholds
    assert w.n_calib_per_stratum["_marginal"] == 120


def test_predict_interval_shapes_and_ordering() -> None:
    model, X_cal, y_cal, X_test, _ = _fit_forrester()
    w = calibrate(model, X_cal, y_cal, alpha=0.10, method="split")
    mu, lower, upper = w.predict_interval(X_test)
    n = X_test.shape[0]
    assert mu.shape == (n,)
    assert lower.shape == (n,)
    assert upper.shape == (n,)
    assert (lower <= mu).all()
    assert (mu <= upper).all()


def test_split_marginal_coverage_near_nominal() -> None:
    """Empirical coverage on a 400-point test set, alpha=0.10 -> nominal 0.90.

    Conformal validity is finite-sample; a 120-calibration / 400-test split
    routinely lands inside [0.85, 0.97]. Loose threshold avoids torch-version
    flakiness.
    """
    model, X_cal, y_cal, X_test, y_test = _fit_forrester()
    w = calibrate(model, X_cal, y_cal, alpha=0.10, method="split")
    cov = w.coverage(X_test, y_test)
    assert 0.83 < cov["_marginal"] < 0.99, f"split coverage={cov}"


def test_cqr_marginal_coverage_near_nominal() -> None:
    model, X_cal, y_cal, X_test, y_test = _fit_forrester()
    w = calibrate(model, X_cal, y_cal, alpha=0.10, method="cqr")
    cov = w.coverage(X_test, y_test)
    assert 0.83 < cov["_marginal"] < 0.99, f"cqr coverage={cov}"


def test_mondrian_strata_register_independently() -> None:
    """Two synthetic strata with different noise produce distinct q_hat values."""
    model, X_cal, y_cal, _, _ = _fit_forrester()

    rng = np.random.default_rng(7)
    strata = rng.choice(["low_alt", "high_alt"], size=len(y_cal))
    y_cal_noisy = y_cal.copy()
    high_mask = strata == "high_alt"
    y_cal_noisy[high_mask] += rng.normal(0.0, 2.0, size=high_mask.sum())

    w = calibrate(
        model,
        X_cal,
        y_cal_noisy,
        alpha=0.10,
        method="split",
        strata=strata,
    )
    assert set(w.strata_thresholds) == {"low_alt", "high_alt"}
    assert w.strata_thresholds["high_alt"] > w.strata_thresholds["low_alt"]


def test_unknown_predict_stratum_raises() -> None:
    model, X_cal, y_cal, X_test, _ = _fit_forrester()
    rng = np.random.default_rng(11)
    strata_cal = rng.choice(["A", "B"], size=len(y_cal))
    strata_test = np.array(["C"] * len(X_test))

    w = calibrate(model, X_cal, y_cal, alpha=0.10, strata=strata_cal)
    with pytest.raises(ValueError, match="not in calibration"):
        w.predict_interval(X_test, strata=strata_test)


def test_alpha_bounds_validated() -> None:
    model, X_cal, y_cal, _, _ = _fit_forrester()
    with pytest.raises(ValueError, match="alpha must be in"):
        calibrate(model, X_cal, y_cal, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        calibrate(model, X_cal, y_cal, alpha=1.5)


def test_unknown_method_rejected() -> None:
    model, X_cal, y_cal, _, _ = _fit_forrester()
    with pytest.raises(ValueError, match="Unknown method"):
        calibrate(model, X_cal, y_cal, method="bogus")  # type: ignore[arg-type]


def test_strata_length_validated_at_calibrate_time() -> None:
    model, X_cal, y_cal, _, _ = _fit_forrester()
    bad_strata = np.array(["A", "B"])
    with pytest.raises(ValueError, match="strata length"):
        calibrate(model, X_cal, y_cal, strata=bad_strata)


def test_determinism_split() -> None:
    model, X_cal, y_cal, X_test, _ = _fit_forrester()
    w1 = calibrate(model, X_cal, y_cal, alpha=0.10, method="split")
    w2 = calibrate(model, X_cal, y_cal, alpha=0.10, method="split")
    _, l1, u1 = w1.predict_interval(X_test)
    _, l2, u2 = w2.predict_interval(X_test)
    np.testing.assert_array_equal(l1, l2)
    np.testing.assert_array_equal(u1, u2)
