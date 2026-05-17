"""Tests confirming the conformal wrapper accepts the Phase 7.2 GPModel."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from .synthetic import forrester_high


@pytest.fixture(autouse=True)
def _deterministic_torch() -> None:
    torch.manual_seed(42)


def test_calibrate_accepts_gp_model() -> None:
    from pulse_research.conformal import ConformalWrapper, calibrate
    from pulse_research.surrogate.gp import train_gp

    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=(60,))
    X = x.reshape(-1, 1)
    y = forrester_high(x)
    gp = train_gp(X, y, seed=42)

    X_cal = rng.uniform(0.0, 1.0, size=(80, 1))
    y_cal = forrester_high(X_cal.ravel())
    wrapper = calibrate(gp, X_cal, y_cal, alpha=0.10, method="split")
    assert isinstance(wrapper, ConformalWrapper)
    assert "_marginal" in wrapper.strata_thresholds


def test_predict_interval_with_gp_model() -> None:
    from pulse_research.conformal import calibrate
    from pulse_research.surrogate.gp import train_gp

    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(60, 1))
    y = forrester_high(X.ravel())
    gp = train_gp(X, y, seed=42)
    X_cal = rng.uniform(0.0, 1.0, size=(100, 1))
    y_cal = forrester_high(X_cal.ravel())
    wrapper = calibrate(gp, X_cal, y_cal, alpha=0.10, method="cqr")

    X_test = rng.uniform(0.0, 1.0, size=(40, 1))
    mu, lower, upper = wrapper.predict_interval(X_test)
    assert mu.shape == (40,)
    assert lower.shape == (40,)
    assert upper.shape == (40,)
    assert np.all(lower <= upper)
