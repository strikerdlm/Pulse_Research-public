"""Tests for the BoTorch SingleTaskMultiFidelityGP wrapper."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pulse_research.surrogate.mfgp import MFGPModel, train_mfgp

from .synthetic import forrester_high, make_forrester_pair


@pytest.fixture(autouse=True)
def _deterministic_torch() -> None:
    torch.manual_seed(42)


def _grid(n: int = 200) -> np.ndarray:
    return np.linspace(0.0, 1.0, n).reshape(-1, 1)


def test_train_returns_mfgp_model() -> None:
    low_X, low_y, high_X, high_y = make_forrester_pair(n_low=30, n_high=6, seed=42)
    model = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    assert isinstance(model, MFGPModel)
    assert model.feature_dim == 1


def test_predict_shapes() -> None:
    low_X, low_y, high_X, high_y = make_forrester_pair(n_low=30, n_high=6, seed=42)
    model = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    X_test = _grid(50)
    mu, sigma = model.predict(X_test, fidelity="high")
    assert mu.shape == (50,)
    assert sigma.shape == (50,)
    assert (sigma >= 0.0).all()


def test_forrester_benchmark_mae_below_threshold() -> None:
    """Posterior MAE on a dense grid is bounded.

    Forrester range is ~22 peak-to-peak. A reasonable MF fit lands below
    2.0; threshold 4.0 avoids flakiness across torch versions.
    """
    low_X, low_y, high_X, high_y = make_forrester_pair(n_low=40, n_high=8, seed=42)
    model = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    X_test = _grid(200)
    mu, _ = model.predict(X_test, fidelity="high")
    truth = forrester_high(X_test.ravel())
    mae = float(np.mean(np.abs(mu - truth)))
    assert mae < 4.0, f"Forrester MAE too high: {mae:.3f}"


def test_mf_beats_sf_on_sparse_high_fid() -> None:
    """MF (30 low + 5 high) beats single-fid GP (5 high only).

    This is the qualitative reason to use MF: the low-fid cohort gives the
    multi-fidelity posterior the shape of the function, and the few high-fid
    samples calibrate the bias correction. A 5-sample single-fid GP cannot
    recover the function from 5 points alone.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    low_X, low_y, high_X, high_y = make_forrester_pair(n_low=30, n_high=5, seed=42)
    X_test = _grid(200)
    truth = forrester_high(X_test.ravel())

    mf_model = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    mu_mf, _ = mf_model.predict(X_test, fidelity="high")
    mae_mf = float(np.mean(np.abs(mu_mf - truth)))

    torch.manual_seed(42)
    sf_train_X = torch.tensor(high_X, dtype=torch.double)
    sf_train_Y = torch.tensor(high_y, dtype=torch.double).reshape(-1, 1)
    sf = SingleTaskGP(sf_train_X, sf_train_Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(sf.likelihood, sf))
    with torch.no_grad():
        sf_post = sf.posterior(torch.tensor(X_test, dtype=torch.double))
        mu_sf = sf_post.mean.squeeze(-1).cpu().numpy()
    mae_sf = float(np.mean(np.abs(mu_sf - truth)))

    assert mae_mf < mae_sf, f"MF MAE {mae_mf:.3f} did not beat SF MAE {mae_sf:.3f}"


def test_seed_determinism() -> None:
    low_X, low_y, high_X, high_y = make_forrester_pair(n_low=20, n_high=5, seed=42)
    X_test = _grid(50)

    m1 = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    mu1, _ = m1.predict(X_test, fidelity="high")

    m2 = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    mu2, _ = m2.predict(X_test, fidelity="high")

    np.testing.assert_allclose(mu1, mu2, rtol=1e-5, atol=1e-5)


def test_residual_mae_returns_nonneg_for_both_fidelities() -> None:
    low_X, low_y, high_X, high_y = make_forrester_pair(n_low=50, n_high=5, seed=42)
    model = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    mae_low = model.residual_mae(low_X, low_y, fidelity="low")
    mae_high = model.residual_mae(high_X, high_y, fidelity="high")
    assert mae_low >= 0.0
    assert mae_high >= 0.0
    assert isinstance(mae_low, float)
    assert isinstance(mae_high, float)


def test_unknown_fidelity_rejected() -> None:
    low_X, low_y, high_X, high_y = make_forrester_pair(n_low=10, n_high=4, seed=42)
    model = train_mfgp(low_X, low_y, high_X, high_y, seed=42)
    with pytest.raises(ValueError, match="fidelity"):
        model.predict(_grid(10), fidelity="medium")  # type: ignore[arg-type]


def test_feature_dim_mismatch_rejected() -> None:
    low_X, low_y, _, _ = make_forrester_pair(n_low=10, n_high=4, seed=42)
    high_X_2d = np.hstack([low_X[:4], low_X[:4]])
    high_y = np.arange(4, dtype=float)
    with pytest.raises(ValueError, match="feature_dim mismatch"):
        train_mfgp(low_X, low_y, high_X_2d, high_y, seed=42)
