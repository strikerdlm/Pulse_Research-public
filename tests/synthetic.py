"""Synthetic multi-fidelity benchmarks. Test-only — not exported from the package.

Forrester (2007) 1-D pair: the field-standard cheap-vs-expensive MF benchmark.
``f_low`` is biased and rescaled relative to ``f_high`` so that a naive
single-fidelity GP trained on a few high-fidelity points loses to a multi-
fidelity GP that consumes a dense low-fidelity cohort.

Reference:
    Forrester A.I.J., Sobester A., Keane A.J. (2007).
    "Multi-fidelity optimization via surrogate modelling."
    Proc R Soc A 463(2088), 3251-3269. doi:10.1098/rspa.2007.1900
"""
from __future__ import annotations

import numpy as np


def forrester_high(x: np.ndarray) -> np.ndarray:
    """Forrester 2007 high-fidelity 1-D function; x in [0, 1]."""
    return (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)


def forrester_low(x: np.ndarray) -> np.ndarray:
    """Forrester 2007 biased low-fidelity surrogate."""
    return 0.5 * forrester_high(x) + 10.0 * (x - 0.5) - 5.0


def make_forrester_pair(
    n_low: int,
    n_high: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(low_X, low_y, high_X, high_y)`` drawn from the Forrester pair.

    Both cohorts use independent uniform-random samples on [0, 1].
    Shapes:
      low_X  ``(n_low, 1)``
      low_y  ``(n_low,)``
      high_X ``(n_high, 1)``
      high_y ``(n_high,)``
    """
    rng = np.random.default_rng(seed)
    low_x = rng.uniform(0.0, 1.0, size=(n_low,))
    high_x = rng.uniform(0.0, 1.0, size=(n_high,))
    return (
        low_x.reshape(-1, 1),
        forrester_low(low_x),
        high_x.reshape(-1, 1),
        forrester_high(high_x),
    )
