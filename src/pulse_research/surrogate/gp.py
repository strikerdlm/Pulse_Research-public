"""BoTorch SingleTaskGP wrapper for Phase 7.2 twin-GP analysis.

Used independently per arm (no multi-fidelity coupling — see spec
docs/superpowers/specs/2026-05-15-phase-7.2-twin-gp-joint-sensitivity-design.md
§1 for why the orthogonal-oracle 7.1b output precludes KOH MF-GP).

The wrapper mirrors :class:`pulse_research.surrogate.mfgp.MFGPModel`'s
predict signature so the existing conformal calibration in
``pulse_research.conformal.wrap`` accepts both via the
:class:`pulse_research.surrogate.types.SurrogateProtocol` Protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


@dataclass
class GPModel:
    """A fitted single-fidelity Gaussian-process surrogate."""

    botorch_model: SingleTaskGP
    feature_dim: int
    dtype: torch.dtype
    device: torch.device
    fit_info: dict[str, Any] = field(default_factory=dict)

    def predict(
        self,
        X: np.ndarray,
        fidelity: Any = None,  # Protocol compat; ignored by single-fid model
        *,
        chunk_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation at each row of ``X``.

        For large ``X`` (e.g., the N_base=8192 Saltelli design with 196 608
        rows), processes the rows in chunks of ``chunk_size``. BoTorch's
        ``model.posterior(X)`` materializes the full joint test-test
        covariance block, which is O(n_test^2) memory; chunking caps each
        block at chunk_size^2.
        """
        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(
                f"X must have shape (n, {self.feature_dim}); got {X.shape}"
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}")
        n = X.shape[0]
        if n <= chunk_size:
            return self._predict_block(X)
        mus = []
        sigmas = []
        for i in range(0, n, chunk_size):
            mu_i, sigma_i = self._predict_block(X[i : i + chunk_size])
            mus.append(mu_i)
            sigmas.append(sigma_i)
        return np.concatenate(mus), np.concatenate(sigmas)

    def _predict_block(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the BoTorch posterior on a single block of rows."""
        X_t = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        self.botorch_model.eval()
        with torch.no_grad():
            posterior = self.botorch_model.posterior(X_t)
            mu = posterior.mean.squeeze(-1).cpu().numpy()
            sigma = (
                posterior.variance.clamp_min(0.0).sqrt().squeeze(-1).cpu().numpy()
            )
        return mu, sigma

    def residual_mae(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fidelity: Any = None,  # Protocol compat; ignored by single-fid model
    ) -> float:
        mu, _ = self.predict(X)
        return float(np.mean(np.abs(mu - y)))


def train_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    nu: float = 2.5,
    jitter: float = 1e-6,
) -> GPModel:
    """Fit a SingleTaskGP with ARD Matern kernel via MLE on `(X, y)`.

    Parameters
    ----------
    X:
        ``(n, d)`` training features. Caller is responsible for selecting
        the *active* axis subset for the arm being fit (e.g. the Pulse
        arm fits on `(fio2_inspired, sao2_baseline)` only, not all 11
        Saltelli axes — see spec §3.2 / Task 11 active-axes rationale).
    y:
        ``(n,)`` training targets.
    seed:
        Torch + numpy RNG seed for deterministic L-BFGS initialization.
    nu:
        Matern smoothness (2.5 default — matches BoTorch's
        ``SingleTaskMultiFidelityGP`` default).
    jitter:
        Initial likelihood noise; helps with near-duplicate rows.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X / y row counts disagree: {X.shape[0]} vs {y.shape[0]}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    dtype = torch.double
    device = torch.device("cpu")

    train_X = torch.as_tensor(X, dtype=dtype, device=device)
    train_Y = torch.as_tensor(y, dtype=dtype, device=device).unsqueeze(-1)

    return _fit(
        train_X=train_X,
        train_Y=train_Y,
        nu=nu,
        jitter=jitter,
        dtype=dtype,
        device=device,
        feature_dim=X.shape[1],
        retry_on_failure=True,
    )


def _fit(
    *,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    nu: float,
    jitter: float,
    dtype: torch.dtype,
    device: torch.device,
    feature_dim: int,
    retry_on_failure: bool,
) -> GPModel:
    covar_module = ScaleKernel(
        MaternKernel(nu=nu, ard_num_dims=feature_dim)
    )
    # Standardize the target (zero mean, unit variance during fit; the
    # transform un-standardizes posterior mean/variance at predict time).
    # Recommended BoTorch practice; matters most for narrow-range targets.
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
    )
    model.likelihood.noise = torch.tensor(jitter, dtype=dtype, device=device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception:  # broad catch is the retry hook
        if not retry_on_failure:
            raise
        # One retry with rougher kernel + larger jitter (spec §3.2).
        return _fit(
            train_X=train_X,
            train_Y=train_Y,
            nu=1.5,
            jitter=1e-4,
            dtype=dtype,
            device=device,
            feature_dim=feature_dim,
            retry_on_failure=False,
        )
    return GPModel(
        botorch_model=model,
        feature_dim=feature_dim,
        dtype=dtype,
        device=device,
        fit_info={
            "nu": nu,
            "jitter": jitter,
            "kernel": "ScaleKernel(Matern(ARD))",
            "outcome_transform": "Standardize",
        },
    )
