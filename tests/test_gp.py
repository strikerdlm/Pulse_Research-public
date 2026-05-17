"""Tests for the BoTorch SingleTaskGP wrapper used by Phase 7.2 twin GPs."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from .synthetic import forrester_high


@pytest.fixture(autouse=True)
def _deterministic_torch() -> None:
    torch.manual_seed(42)


def _make_forrester_single_fid(
    n: int = 60, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n,))
    return x.reshape(-1, 1), forrester_high(x)


def test_train_returns_gp_model() -> None:
    from pulse_research.surrogate.gp import GPModel, train_gp

    X, y = _make_forrester_single_fid(n=40, seed=42)
    model = train_gp(X, y, seed=42)
    assert isinstance(model, GPModel)
    assert model.feature_dim == 1


def test_predict_shapes() -> None:
    from pulse_research.surrogate.gp import train_gp

    X, y = _make_forrester_single_fid(n=40, seed=42)
    model = train_gp(X, y, seed=42)
    X_test = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    mu, sigma = model.predict(X_test)
    assert mu.shape == (50,)
    assert sigma.shape == (50,)
    assert np.all(sigma >= 0.0)


def test_forrester_mae_below_bound() -> None:
    """Trained GP recovers the Forrester high-fid function to MAE < 0.5 on a 200-pt grid."""
    from pulse_research.surrogate.gp import train_gp

    X, y = _make_forrester_single_fid(n=60, seed=42)
    model = train_gp(X, y, seed=42)
    X_test = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    y_test = forrester_high(X_test.ravel())
    mu, _ = model.predict(X_test)
    mae = float(np.mean(np.abs(mu - y_test)))
    assert mae < 0.5, f"Forrester MAE bound violated: {mae:.4f}"


def test_determinism_same_seed() -> None:
    """Two fits on identical inputs at the same seed produce identical posterior means."""
    from pulse_research.surrogate.gp import train_gp

    X, y = _make_forrester_single_fid(n=40, seed=42)
    X_test = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    torch.manual_seed(42)
    mu_a, _ = train_gp(X, y, seed=42).predict(X_test)
    torch.manual_seed(42)
    mu_b, _ = train_gp(X, y, seed=42).predict(X_test)
    np.testing.assert_allclose(mu_a, mu_b, rtol=1e-10, atol=1e-12)


def test_retry_on_fit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If first fit raises, train_gp retries with nu=1.5 and jitter=1e-4."""
    from pulse_research.surrogate import gp as gp_module

    X, y = _make_forrester_single_fid(n=30, seed=42)

    call_log: list[float] = []
    real_fit: Any = gp_module.fit_gpytorch_mll  # type: ignore[attr-defined]

    def fake_fit(mll: Any, *args: Any, **kwargs: Any) -> None:
        # Read the learned kernel's nu via the underlying MaternKernel.
        kernel = mll.model.covar_module
        nu = float(kernel.base_kernel.nu)
        call_log.append(nu)
        if len(call_log) == 1:
            raise RuntimeError("synthetic L-BFGS failure")
        real_fit(mll, *args, **kwargs)

    monkeypatch.setattr(gp_module, "fit_gpytorch_mll", fake_fit)
    model = gp_module.train_gp(X, y, seed=42)
    assert len(call_log) == 2, f"expected exactly one retry; got {call_log!r}"
    assert call_log[0] == 2.5
    assert call_log[1] == 1.5
    assert model.fit_info["nu"] == 1.5


def test_handles_near_duplicate_rows() -> None:
    """Two rows differing by 1e-7 in one axis don't trip a LinAlgError."""
    from pulse_research.surrogate.gp import train_gp

    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(20, 2))
    # Inject a near-duplicate pair.
    X = np.vstack([X, X[0:1] + np.array([[1e-7, 0.0]])])
    y = (X[:, 0] - 0.5) ** 2 + (X[:, 1] - 0.5) ** 2
    model = train_gp(X, y, seed=42)
    mu, _ = model.predict(X)
    assert mu.shape == (X.shape[0],)
    assert np.all(np.isfinite(mu))
