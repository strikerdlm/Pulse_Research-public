"""11-axis Saltelli-Sobol design for the CGEM-Pulse hypoxia surrogate.

The 9 existing CGEM axes are extended with two hypoxia-relevant axes
(``fio2_inspired``, ``sao2_baseline``) per
`docs/research/deep_dive_1_cgem_pulse_multifidelity_hypoxia.md` section 2.

Uses ``SALib.sample.sobol.sample`` (canonical post-1.4 path; the legacy
``saltelli.sample`` import is deprecated). Power-of-two ``N`` is enforced
because Saltelli's A/B/AB construction is only Sobol-valid for those sizes.
"""
from __future__ import annotations

from typing import Final

import numpy as np
from SALib.sample import sobol as salib_sobol

AXIS_NAMES: Final[list[str]] = [
    "gz_peak",
    "gz_onset_rate",
    "seat_tilt_deg",
    "anti_g_strain",
    "pilot_weight_kg",
    "pilot_height_cm",
    "pilot_age_y",
    "baseline_vo2max",
    "baseline_map_mmhg",
    "fio2_inspired",
    "sao2_baseline",
]

AXIS_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "gz_peak": (1.0, 9.0),
    "gz_onset_rate": (0.1, 6.0),
    "seat_tilt_deg": (0.0, 30.0),
    "anti_g_strain": (0.0, 1.0),
    "pilot_weight_kg": (50.0, 110.0),
    "pilot_height_cm": (155.0, 200.0),
    "pilot_age_y": (20.0, 55.0),
    "baseline_vo2max": (30.0, 70.0),
    "baseline_map_mmhg": (70.0, 110.0),
    "fio2_inspired": (0.15, 1.00),
    "sao2_baseline": (0.80, 1.00),
}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def build_design(n_base: int = 1024, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Saltelli-Sobol design over the 11-axis CGEM-Pulse space.

    Parameters
    ----------
    n_base:
        Base sample size ``N``. Must be a power of two (Saltelli A/B/AB
        construction requirement). Total rows = ``N * (2k + 2)`` where
        ``k = 11`` → ``N * 24``.
    seed:
        RNG seed. Default ``42`` is the canonical seed used throughout
        ``cgem_synthetic_v1``.

    Returns
    -------
    X:
        ``(n_base * 24, 11)`` Saltelli sample.
    names:
        Axis names in column order (same as :data:`AXIS_NAMES`).
    """
    if not _is_pow2(n_base):
        raise ValueError(f"n_base must be a power of two, got {n_base}")

    problem = {
        "num_vars": len(AXIS_NAMES),
        "names": AXIS_NAMES,
        "bounds": [list(AXIS_BOUNDS[name]) for name in AXIS_NAMES],
    }
    sample: np.ndarray = salib_sobol.sample(
        problem,
        N=n_base,
        calc_second_order=True,
        seed=seed,
    )
    return sample, list(AXIS_NAMES)
