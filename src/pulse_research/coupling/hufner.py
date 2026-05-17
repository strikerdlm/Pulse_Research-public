"""Hüfner CaO2 ratio coupling for Phase 7.3.

Pure functions over numpy arrays. No GP coupling here — the orchestrator
in ``scripts/run_phase7_3.py`` evaluates both GP posteriors and passes
the (mean, sigma) tuples here for combination and MC propagation.

The coupling form (from Phase 7.2 spec §11, locked in Phase 7.3 spec §3.2):

    corrected_time(x) = time_cgem(x)
                      * CaO2(SaO2 = O2_pulse(x), Hb, PaO2(FiO2, altitude=0))
                      / CaO2(SaO2 = 0.97, Hb, PaO2_sea_level)

PaO2 derives from the simplified alveolar gas equation at sea level:
    PaO2 = FiO2 * (760 - 47) - 40 / 0.8 = FiO2 * 713 - 50

No fitted parameter; the entire scientific edifice rests on the Hüfner
formula (CaO2 = 1.34 * Hb * SaO2 + 0.003 * PaO2) and the canonical
PaO2 derivation. See Phase 7.3 spec §1 hard rules.
"""
from __future__ import annotations

import numpy as np

# Published constants. Sources verified against
# /root/repos/HumanPerformanceCalcs/frontend/src/calculators/atmosphere.ts
# (commit cc35310 in pulse_research, 2026-05-15).
_HB_BASELINE_G_DL = 14.5
_SAO2_REFERENCE_FRACTION = 0.97
_HUFNER_CONSTANT = 1.34  # mL O2 / g Hb
_DISSOLVED_O2_PER_MMHG = 0.003  # mL O2 / dL / mmHg
_SEA_LEVEL_PRESSURE_MMHG = 760.0
_WATER_VAPOR_PRESSURE_37C_MMHG = 47.0
_PACO2_BASELINE_MMHG = 40.0
_RQ_BASELINE = 0.8


def _pao2_at_fio2(fio2: float | np.ndarray) -> float | np.ndarray:
    """Simplified alveolar gas equation at sea level.

    PaO2 = FiO2 * (P_bar - P_H2O) - PaCO2 / RQ.
    Vectorized over numpy arrays; pure arithmetic.
    """
    pio2 = fio2 * (_SEA_LEVEL_PRESSURE_MMHG - _WATER_VAPOR_PRESSURE_37C_MMHG)
    return pio2 - _PACO2_BASELINE_MMHG / _RQ_BASELINE


def cao2_at(
    fio2: float,
    sao2_fraction: float,
    *,
    hb_g_dl: float = _HB_BASELINE_G_DL,
) -> float:
    """CaO2 (mL O2 / dL) at given FiO2 + SaO2 (sea-level reference)."""
    pao2 = _pao2_at_fio2(float(fio2))
    return float(
        _HUFNER_CONSTANT * hb_g_dl * sao2_fraction
        + _DISSOLVED_O2_PER_MMHG * pao2
    )


def cao2_reference(*, hb_g_dl: float = _HB_BASELINE_G_DL) -> float:
    """Sea-level normoxic reference CaO2 (FiO2=0.21, SaO2=0.97)."""
    return cao2_at(0.21, _SAO2_REFERENCE_FRACTION, hb_g_dl=hb_g_dl)


def corrected_time(
    time_to_gloc_s: np.ndarray,
    sao2_fraction: np.ndarray,
    fio2: np.ndarray,
    *,
    hb_g_dl: float = _HB_BASELINE_G_DL,
) -> np.ndarray:
    """corrected_time = time * CaO2(O2, FiO2) / CaO2_ref.

    Vectorized element-wise over the three inputs. Raises on shape mismatch.
    """
    t = np.asarray(time_to_gloc_s, dtype=float)
    s = np.asarray(sao2_fraction, dtype=float)
    f = np.asarray(fio2, dtype=float)
    if t.shape != s.shape or t.shape != f.shape:
        raise ValueError(
            f"shape mismatch: time {t.shape}, sao2 {s.shape}, fio2 {f.shape}"
        )
    pao2 = _pao2_at_fio2(f)
    cao2 = _HUFNER_CONSTANT * hb_g_dl * s + _DISSOLVED_O2_PER_MMHG * pao2
    return t * cao2 / cao2_reference(hb_g_dl=hb_g_dl)


def mc_propagate(
    *,
    time_mu: np.ndarray,
    time_sigma: np.ndarray,
    o2_mu: np.ndarray,
    o2_sigma: np.ndarray,
    fio2: np.ndarray,
    n_mc: int = 2048,
    seed: int = 42,
    alpha: float = 0.10,
    hb_g_dl: float = _HB_BASELINE_G_DL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte-Carlo propagate CGEM and Pulse GP posteriors through Hüfner.

    Independence assumption: the two posteriors are drawn independently
    because the GPs were fit on different oracle outputs in Phase 7.2.
    This is the correct joint sampling for orthogonal-oracle 7.1b data.

    Returns
    -------
    (mu_corrected, lower, upper):
        Each shape ``(n,)``. ``mu_corrected`` is the sample mean of
        the propagated corrected_time draws. ``lower`` / ``upper`` are
        the alpha/2 and 1-alpha/2 empirical quantiles.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if n_mc <= 0:
        raise ValueError(f"n_mc must be positive; got {n_mc}")
    rng = np.random.default_rng(seed)
    n = time_mu.shape[0]
    for name, arr in (
        ("time_sigma", time_sigma), ("o2_mu", o2_mu),
        ("o2_sigma", o2_sigma), ("fio2", fio2),
    ):
        if arr.shape != (n,):
            raise ValueError(
                f"all per-row inputs must share shape (n,); {name} has {arr.shape}"
            )
    # (n_mc, n) sample banks; independent across the two oracles.
    z_time = rng.standard_normal(size=(n_mc, n))
    z_o2 = rng.standard_normal(size=(n_mc, n))
    time_samples = time_mu + time_sigma * z_time
    o2_samples = np.clip(o2_mu + o2_sigma * z_o2, 0.0, 1.0)
    fio2_b = np.broadcast_to(fio2, (n_mc, n))
    corrected_samples = corrected_time(
        time_samples, o2_samples, fio2_b, hb_g_dl=hb_g_dl,
    )
    mu_corr = corrected_samples.mean(axis=0)
    q_low = alpha / 2.0
    q_high = 1.0 - alpha / 2.0
    lower = np.quantile(corrected_samples, q_low, axis=0)
    upper = np.quantile(corrected_samples, q_high, axis=0)
    return mu_corr, lower, upper
