"""Duck-typed Protocols shared across surrogate variants.

The conformal calibration wrapper in ``pulse_research.conformal.wrap`` is
generic over the surrogate family: it only needs ``predict(X, fidelity=...)
-> (mu, sigma)`` and a ``feature_dim`` integer attribute. By depending on a
``Protocol`` here rather than on the concrete ``MFGPModel`` dataclass, the
same wrapper accepts the Phase 7.2 ``GPModel`` (single-fidelity) without
duplication.

The ``fidelity`` argument is part of the Protocol because ``MFGPModel``
requires it; single-fidelity implementations accept and ignore it via
``fidelity: Any = None``.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SurrogateProtocol(Protocol):
    """Minimum interface conformal calibration needs from a surrogate."""

    feature_dim: int

    def predict(
        self,
        X: np.ndarray,
        fidelity: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mu, sigma)`` posterior arrays of shape ``(n,)``."""
        ...
