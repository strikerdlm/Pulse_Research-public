"""Split and locally-adaptive conformal calibration around an MFGPModel.

Implements:

- **Split conformal** (absolute residual).
- **CQR-style locally adaptive** (sigma-scaled): uses the GP's posterior sigma
  as the difficulty estimator, producing locally adaptive intervals analogous
  to Romano 2019 CQR without requiring a separate quantile regressor.
- **Mondrian stratification** over an arbitrary categorical label (intended
  use: altitude band 0-5k / 5k-15k / 15k-25k ft).

The implementation is direct numpy/scipy (~50 lines of math); MAPIE is not
pulled into the dependency graph at this phase. A Phase-3.5 ``to_mapie()``
adapter is a small follow-up if benchmark comparisons are needed.

References:
    Vovk V., Gammerman A., Shafer G. (2005). "Algorithmic Learning in a
    Random World." Springer. — split conformal foundation.

    Romano Y., Patterson E., Candes E.J. (2019). "Conformalized Quantile
    Regression." NeurIPS 32. arXiv:1905.03222 — CQR original.

    Angelopoulos A.N., Bates S. (2023). "Conformal Prediction: A Gentle
    Introduction." Found Trends Mach Learn 16(4):494-591.
    doi:10.1561/2200000101.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from pulse_research.surrogate.mfgp import Fidelity
from pulse_research.surrogate.types import SurrogateProtocol

_MARGINAL_KEY = "_marginal"
_SIGMA_FLOOR = 1e-8

ConformalMethod = Literal["split", "cqr"]


@dataclass
class ConformalWrapper:
    """Fitted conformal calibration around a surrogate model.

    Attributes
    ----------
    surrogate_model:
        The underlying surrogate; calibration was performed against its
        posterior mean (and sigma, for ``"cqr"``). Accepts any object that
        satisfies :class:`pulse_research.surrogate.types.SurrogateProtocol`
        — both :class:`MFGPModel` (Phase 2/3) and :class:`GPModel`
        (Phase 7.2) qualify.
    method:
        ``"split"`` (absolute residual) or ``"cqr"`` (sigma-scaled).
    alpha:
        Miscoverage rate; nominal coverage is ``1 - alpha``.
    fidelity:
        Surrogate fidelity surface used for calibration.
    strata_thresholds:
        Map stratum label -> ``q_hat``. For unstratified calibration, the only
        key is ``"_marginal"``.
    n_calib_per_stratum:
        Map stratum label -> calibration sample count.
    """

    surrogate_model: SurrogateProtocol
    method: ConformalMethod
    alpha: float
    fidelity: Fidelity
    strata_thresholds: dict[str, float]
    n_calib_per_stratum: dict[str, int]

    @property
    def mfgp_model(self) -> SurrogateProtocol:
        """Backward-compat alias retained because Phase 2/3 tests use this name."""
        return self.surrogate_model

    def _stratum_keys(self, strata: np.ndarray | None, n: int) -> np.ndarray:
        if strata is None:
            return np.full(n, _MARGINAL_KEY, dtype=object)
        if len(strata) != n:
            raise ValueError(f"strata length {len(strata)} != X.shape[0] {n}")
        return np.asarray([str(s) for s in strata], dtype=object)

    def predict_interval(
        self,
        X: np.ndarray,
        *,
        strata: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior mean and conformal lower / upper bounds for each row of X.

        Predict-time strata labels MUST already appear in
        ``self.strata_thresholds``; an unseen label raises ``ValueError``.
        """
        mu, sigma = self.surrogate_model.predict(X, fidelity=self.fidelity)
        labels = self._stratum_keys(strata, len(mu))

        unknown = sorted(set(labels) - set(self.strata_thresholds))
        if unknown:
            raise ValueError(
                f"predict-time strata {unknown!r} not in calibration set "
                f"{sorted(self.strata_thresholds)!r}"
            )

        q = np.asarray(
            [self.strata_thresholds[label] for label in labels], dtype=float
        )
        if self.method == "split":
            lower = mu - q
            upper = mu + q
        else:  # "cqr"
            scale = np.maximum(sigma, _SIGMA_FLOOR)
            lower = mu - q * scale
            upper = mu + q * scale
        return mu, lower, upper

    def coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        strata: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Empirical coverage proportion per stratum (or ``"_marginal"``)."""
        _, lower, upper = self.predict_interval(X_test, strata=strata)
        covered = (y_test >= lower) & (y_test <= upper)
        labels = self._stratum_keys(strata, len(y_test))
        result: dict[str, float] = {}
        for label in self.strata_thresholds:
            mask = labels == label
            if mask.any():
                result[label] = float(covered[mask].mean())
        return result


def _conformal_quantile_level(n: int, alpha: float) -> float:
    """Finite-sample-valid quantile level for split-conformal q_hat."""
    return float(np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0))


def _residuals(
    method: ConformalMethod,
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    if method == "split":
        return np.asarray(np.abs(y - mu), dtype=float)
    if method == "cqr":
        scale = np.maximum(sigma, _SIGMA_FLOOR)
        return np.asarray(np.abs(y - mu) / scale, dtype=float)
    raise ValueError(f"Unknown method {method!r}; expected 'split' or 'cqr'.")


def calibrate(
    surrogate_model: SurrogateProtocol,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    alpha: float = 0.10,
    method: ConformalMethod = "split",
    fidelity: Fidelity = "high",
    strata: np.ndarray | None = None,
) -> ConformalWrapper:
    """Compute calibration residuals and per-stratum ``q_hat`` thresholds.

    Parameters
    ----------
    surrogate_model:
        A fitted surrogate satisfying
        :class:`pulse_research.surrogate.types.SurrogateProtocol` — either
        a Phase 2/3 :class:`MFGPModel` or a Phase 7.2 :class:`GPModel`.
    X_calib, y_calib:
        Held-out calibration cohort.
    alpha:
        Miscoverage rate (nominal coverage = ``1 - alpha``); must be in
        ``(0, 1)``.
    method:
        ``"split"`` (absolute residual) or ``"cqr"`` (sigma-scaled).
    fidelity:
        Which surrogate fidelity to calibrate against.
    strata:
        Optional categorical labels of shape ``(n_calib,)`` for Mondrian
        stratification; each unique label gets its own ``q_hat``.

    Returns
    -------
    ConformalWrapper
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if y_calib.ndim != 1:
        raise ValueError("y_calib must be 1-D")
    if X_calib.shape[0] != y_calib.shape[0]:
        raise ValueError("X_calib and y_calib row counts disagree")

    mu, sigma = surrogate_model.predict(X_calib, fidelity=fidelity)
    s = _residuals(method, mu, sigma, y_calib)

    if strata is None:
        labels = np.full(len(y_calib), _MARGINAL_KEY, dtype=object)
    else:
        if len(strata) != len(y_calib):
            raise ValueError(
                f"strata length {len(strata)} != y_calib length {len(y_calib)}"
            )
        labels = np.asarray([str(t) for t in strata], dtype=object)

    thresholds: dict[str, float] = {}
    n_per_stratum: dict[str, int] = {}
    for label in np.unique(labels):
        mask = labels == label
        s_stratum = s[mask]
        n_s = int(mask.sum())
        if n_s == 0:
            continue
        level = _conformal_quantile_level(n_s, alpha)
        thresholds[str(label)] = float(np.quantile(s_stratum, level))
        n_per_stratum[str(label)] = n_s

    return ConformalWrapper(
        surrogate_model=surrogate_model,
        method=method,
        alpha=alpha,
        fidelity=fidelity,
        strata_thresholds=thresholds,
        n_calib_per_stratum=n_per_stratum,
    )
