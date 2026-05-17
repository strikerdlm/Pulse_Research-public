"""BoTorch SingleTaskMultiFidelityGP wrapper.

The wrapper owns the fidelity-column convention: the last column of X is the
fidelity flag, with values in ``{0.0, 1.0}`` for ``{low, high}``. Callers pass
plain feature matrices and a ``fidelity="low" | "high"`` string; the wrapper
appends the fidelity column before querying BoTorch so the convention never
leaks into call sites.

KOH-equivalent kernel: ``LinearTruncatedFidelityKernel(nu=2.5,
linear_truncated=True)``. For two fidelity levels, this matches the
Kennedy-O'Hagan (2000) discrete linear scale+bias decomposition.

References:
    Kennedy MC, O'Hagan A (2000). "Predicting the output from a complex
    computer code when fast approximations are available." Biometrika
    87(1):1-13.

    Wu J, Toscano-Palmerin S, Frazier PI, Wilson AG (2019). "Practical
    Multi-fidelity Bayesian Optimization for Hyperparameter Tuning." UAI 2019.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskMultiFidelityGP
from gpytorch.mlls import ExactMarginalLogLikelihood

_LOW = 0.0
_HIGH = 1.0

Fidelity = Literal["low", "high"]


@dataclass
class MFGPModel:
    """A fitted multi-fidelity GP plus the metadata predict() needs.

    Attributes
    ----------
    botorch_model:
        The underlying ``SingleTaskMultiFidelityGP``.
    feature_dim:
        Number of physical feature columns (does NOT include the fidelity flag).
    dtype:
        Tensor dtype used during training (typically ``torch.double``).
    device:
        Device tensors live on (typically CPU for now).
    """

    botorch_model: SingleTaskMultiFidelityGP
    feature_dim: int
    dtype: torch.dtype
    device: torch.device

    def _fidelity_value(self, fidelity: Fidelity) -> float:
        if fidelity == "low":
            return _LOW
        if fidelity == "high":
            return _HIGH
        raise ValueError(
            f"Unknown fidelity {fidelity!r}; expected 'low' or 'high'."
        )

    def _augment(self, X: np.ndarray, fidelity: Fidelity) -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(
                f"X must have shape (n, {self.feature_dim}); got {X.shape}"
            )
        fid = self._fidelity_value(fidelity)
        fid_col = np.full((X.shape[0], 1), fid)
        X_aug = np.hstack([X, fid_col])
        return torch.as_tensor(X_aug, dtype=self.dtype, device=self.device)

    def predict(
        self,
        X: np.ndarray,
        fidelity: Fidelity = "high",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at the given fidelity.

        Parameters
        ----------
        X:
            ``(n, feature_dim)`` query matrix WITHOUT the fidelity column.
        fidelity:
            ``"low"`` or ``"high"``.

        Returns
        -------
        mu:
            ``(n,)`` posterior mean.
        sigma:
            ``(n,)`` posterior standard deviation.
        """
        X_aug = self._augment(X, fidelity)
        self.botorch_model.eval()
        with torch.no_grad():
            posterior = self.botorch_model.posterior(X_aug)
            mu = posterior.mean.squeeze(-1).cpu().numpy()
            sigma = (
                posterior.variance.clamp_min(0.0).sqrt().squeeze(-1).cpu().numpy()
            )
        return mu, sigma

    def residual_mae(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fidelity: Fidelity,
    ) -> float:
        """Mean absolute residual between the posterior mean and ``y`` at this fidelity."""
        mu, _ = self.predict(X, fidelity=fidelity)
        return float(np.mean(np.abs(mu - y)))


def _stack_with_fidelity(
    low_X: np.ndarray,
    low_y: np.ndarray,
    high_X: np.ndarray,
    high_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if low_X.ndim != 2 or high_X.ndim != 2:
        raise ValueError("low_X and high_X must be 2-D")
    if low_X.shape[1] != high_X.shape[1]:
        raise ValueError(
            f"feature_dim mismatch: low_X has {low_X.shape[1]} cols, "
            f"high_X has {high_X.shape[1]}"
        )
    low_aug = np.hstack([low_X, np.full((low_X.shape[0], 1), _LOW)])
    high_aug = np.hstack([high_X, np.full((high_X.shape[0], 1), _HIGH)])
    X = np.vstack([low_aug, high_aug])
    y = np.concatenate([low_y, high_y]).reshape(-1, 1)
    return X, y


def train_mfgp(
    low_X: np.ndarray,
    low_y: np.ndarray,
    high_X: np.ndarray,
    high_y: np.ndarray,
    *,
    seed: int = 42,
    nu: float = 2.5,
) -> MFGPModel:
    """Fit a Kennedy-O'Hagan-equivalent two-fidelity GP.

    Parameters
    ----------
    low_X, low_y, high_X, high_y:
        Paired cohorts. Both ``X`` matrices must share ``feature_dim``; the
        fidelity column is appended internally.
    seed:
        Torch RNG seed; controls L-BFGS init determinism.
    nu:
        Matern smoothness for the base kernel (2.5 is the BoTorch default).

    Returns
    -------
    MFGPModel
    """
    if low_y.ndim != 1 or high_y.ndim != 1:
        raise ValueError("low_y and high_y must be 1-D")
    if low_X.shape[0] != low_y.shape[0] or high_X.shape[0] != high_y.shape[0]:
        raise ValueError("X / y row counts disagree")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dtype = torch.double
    device = torch.device("cpu")

    X_np, y_np = _stack_with_fidelity(low_X, low_y, high_X, high_y)
    train_X = torch.as_tensor(X_np, dtype=dtype, device=device)
    train_Y = torch.as_tensor(y_np, dtype=dtype, device=device)

    feature_dim = low_X.shape[1]
    fidelity_col = feature_dim

    model = SingleTaskMultiFidelityGP(
        train_X=train_X,
        train_Y=train_Y,
        data_fidelities=[fidelity_col],
        linear_truncated=True,
        nu=nu,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return MFGPModel(
        botorch_model=model,
        feature_dim=feature_dim,
        dtype=dtype,
        device=device,
    )
