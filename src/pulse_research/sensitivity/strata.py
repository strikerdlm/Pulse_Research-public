"""Mondrian-strata binning for conformal calibration over altitude tiers.

The three-tier mapping mirrors the Pulse Engine environment-file tiers used
by ``pulse_research.api.pulse_glue.render_pulse_script``: FiO2 >= 0.20 is
normobaric (Standard.json); 0.16 <= FiO2 < 0.20 is Hypobaric3000m.json;
FiO2 < 0.16 is Hypobaric4000m.json. The same binning is applied to the
CGEM arm so per-tier conformal coverage compares like for like across arms.
"""
from __future__ import annotations

import numpy as np

_BOUNDARIES = (0.20, 0.16)
_LABELS = ("normo", "hypo3000", "hypo4000")


def fio2_tier(fio2: np.ndarray) -> np.ndarray:
    """Vectorized 3-tier label per row of ``fio2``.

    Returns a 1-D array of string labels with the same length as ``fio2``.
    Raises ``ValueError`` if ``fio2`` is not 1-D.
    """
    if fio2.ndim != 1:
        raise ValueError(f"fio2 must be 1-D; got shape {fio2.shape}")
    return np.where(
        fio2 >= _BOUNDARIES[0],
        _LABELS[0],
        np.where(fio2 >= _BOUNDARIES[1], _LABELS[1], _LABELS[2]),
    )
