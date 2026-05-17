"""Saltelli-Sobol analyzer wrapper.

Consumes an aligned 1-D output vector from a design built by
:func:`pulse_research.sensitivity.sobol_design.build_design` and returns the
first-order (S1), total-order (ST), and optional second-order (S2) Sobol
indices with bootstrap confidence half-widths.

Raises on NaN inputs by design: Saltelli's A/B/AB construction is not robust
to dropped rows, so silent imputation would produce wrong indices.

References:
    Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S
    (2010). Variance based sensitivity analysis of model output. Design and
    estimator for the total sensitivity index. Comp Phys Comm 181(2):259-270.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from SALib.analyze import sobol as salib_sobol_analyze

from pulse_research.sensitivity.sobol_design import AXIS_BOUNDS, AXIS_NAMES


@dataclass
class SobolIndices:
    """Saltelli-Sobol indices with bootstrap confidence intervals."""

    names: list[str]
    S1: np.ndarray
    S1_conf: np.ndarray
    ST: np.ndarray
    ST_conf: np.ndarray
    S2: np.ndarray | None
    S2_conf: np.ndarray | None
    n_resamples: int  # provenance: echoed from the analyze_design() call


def _default_problem() -> dict[str, Any]:
    return {
        "num_vars": len(AXIS_NAMES),
        "names": list(AXIS_NAMES),
        "bounds": [list(AXIS_BOUNDS[name]) for name in AXIS_NAMES],
    }


def analyze_design(
    outputs: np.ndarray,
    *,
    num_resamples: int = 500,
    seed: int = 42,
    calc_second_order: bool = True,
    problem: dict[str, Any] | None = None,
) -> SobolIndices:
    """Run SALib's Saltelli analyzer over an aligned output vector.

    Parameters
    ----------
    outputs:
        ``(n_rows,)`` simulator outputs aligned with a Saltelli design built
        by :func:`pulse_research.sensitivity.sobol_design.build_design`. Row
        order MUST match SALib's A/B/AB convention; ``n_rows = N * (2k + 2)``
        for ``k`` variables when ``calc_second_order=True`` and ``N * (k + 2)``
        otherwise.
    num_resamples:
        Bootstrap resamples for the SALib confidence half-widths.
    seed:
        RNG seed forwarded to ``SALib.analyze.sobol.analyze`` (controls the
        bootstrap draws, not the design).
    calc_second_order:
        Must match the value passed to ``build_design`` (or the matching
        ``SALib.sample.sobol.sample`` call) — otherwise the analyzer
        receives the wrong row count and silently mis-computes the indices.
    problem:
        Optional SALib problem dict. Defaults to the 11-axis Pulse_Research
        space (``AXIS_NAMES`` x ``AXIS_BOUNDS``).

    Raises
    ------
    ValueError
        If ``outputs`` is not 1-D or contains any NaN.
    """
    if outputs.ndim != 1:
        raise ValueError(f"outputs must be 1-D; got shape {outputs.shape}")
    if np.any(np.isnan(outputs)):
        raise ValueError(
            "outputs contains NaN; Saltelli analysis requires clean inputs"
        )

    prob = problem if problem is not None else _default_problem()

    Si = salib_sobol_analyze.analyze(
        prob,
        outputs,
        calc_second_order=calc_second_order,
        num_resamples=num_resamples,
        conf_level=0.95,
        seed=seed,
        print_to_console=False,
    )

    S2 = np.asarray(Si["S2"]) if calc_second_order else None
    S2_conf = np.asarray(Si["S2_conf"]) if calc_second_order else None

    return SobolIndices(
        names=list(prob["names"]),
        S1=np.asarray(Si["S1"]),
        S1_conf=np.asarray(Si["S1_conf"]),
        ST=np.asarray(Si["ST"]),
        ST_conf=np.asarray(Si["ST_conf"]),
        S2=S2,
        S2_conf=S2_conf,
        n_resamples=num_resamples,
    )


def st_stability(indices: SobolIndices) -> float:
    """Bootstrap-CI-relative-width stability for the ST indices.

    Returns ``1 - max_i(ST_conf_i / ST_i)`` over features with
    ``ST_i > 1e-9``; features below the floor are excluded so the metric is
    not dominated by ratios like ``ST_conf / ~0`` for axes the simulator
    barely activates. Returns ``1.0`` when no feature is active.

    Per Sarrazin et al. 2016, the bootstrap CI relative width is the
    canonical convergence diagnostic for variance-based sensitivity indices.
    ``ST_conf`` here is the 95% half-width returned by
    :func:`analyze_design`.

    Parameters
    ----------
    indices:
        A :class:`SobolIndices` instance produced by
        :func:`analyze_design`. The function does not re-run the analyzer.

    Returns
    -------
    float
        Stability score in ``(-∞, 1]``. The score is negative when the
        bootstrap CI half-width exceeds the point estimate for at least
        one active feature — a sign that ``num_resamples`` and/or the
        Saltelli ``N`` are too small for the design to have converged.

    Notes
    -----
    The earlier "seed sweep CoV" formulation was vacuous: SALib's ``seed``
    parameter only feeds the bootstrap RNG, not the ST point estimate, so
    repeated calls on identical ``outputs`` produce identical ST vectors.
    A correct point-estimate sweep would require re-sampling the Saltelli
    design per seed and re-running the simulator — out of scope for the
    synthetic-data-only IJNMBE paper.
    """
    active = indices.ST > 1e-9
    if not np.any(active):
        return 1.0
    ratios = indices.ST_conf[active] / indices.ST[active]
    return float(1.0 - np.max(ratios))
