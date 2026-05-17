"""Tests for the Hüfner CaO2 ratio coupling used in Phase 7.3."""
from __future__ import annotations

import numpy as np
import pytest


def test_cao2_reference_canonical() -> None:
    """At baseline (Hb=14.5, SaO2=0.97, sea-level FiO2=0.21), CaO2 ~ 19.5 mL O2/dL."""
    from pulse_research.coupling.hufner import cao2_reference

    cao2 = cao2_reference(hb_g_dl=14.5)
    # Bound term: 1.34 * 14.5 * 0.97 = 18.85; dissolved: 0.003 * 100 = 0.30
    assert 18.5 <= cao2 <= 19.5


def test_corrected_time_identity_at_reference() -> None:
    """corrected_time(t, 0.97, 0.21) ~= t (ratio is ~1.0)."""
    from pulse_research.coupling.hufner import corrected_time

    t_in = np.array([1.0, 4.5, 8.7])
    sao2_in = np.full(3, 0.97)
    fio2_in = np.full(3, 0.21)
    t_out = corrected_time(t_in, sao2_in, fio2_in)
    np.testing.assert_allclose(t_out, t_in, rtol=0.01)


def test_corrected_time_drops_with_sao2() -> None:
    """Lower SaO2 yields strictly lower corrected_time at fixed FiO2 and t."""
    from pulse_research.coupling.hufner import corrected_time

    t_in = np.full(3, 5.0)
    fio2_in = np.full(3, 0.21)
    sao2_high = np.full(3, 0.97)
    sao2_low = np.full(3, 0.80)
    t_high = corrected_time(t_in, sao2_high, fio2_in)
    t_low = corrected_time(t_in, sao2_low, fio2_in)
    assert np.all(t_low < t_high)


def test_corrected_time_shape_mismatch_raises() -> None:
    from pulse_research.coupling.hufner import corrected_time

    with pytest.raises(ValueError, match="shape mismatch"):
        corrected_time(np.array([1.0, 2.0]), np.array([0.97]), np.array([0.21]))


def test_mc_propagate_shape_and_bounds() -> None:
    """mc_propagate returns (n,) arrays; lower <= mu <= upper."""
    from pulse_research.coupling.hufner import mc_propagate

    n = 10
    rng = np.random.default_rng(0)
    time_mu = rng.uniform(1.0, 8.0, size=n)
    time_sigma = rng.uniform(0.1, 0.5, size=n)
    o2_mu = rng.uniform(0.88, 0.97, size=n)
    o2_sigma = rng.uniform(0.005, 0.02, size=n)
    fio2 = rng.uniform(0.15, 1.0, size=n)
    mu, lower, upper = mc_propagate(
        time_mu=time_mu, time_sigma=time_sigma,
        o2_mu=o2_mu, o2_sigma=o2_sigma,
        fio2=fio2, n_mc=512, seed=42, alpha=0.10,
    )
    assert mu.shape == (n,)
    assert lower.shape == (n,)
    assert upper.shape == (n,)
    assert np.all(lower <= mu)
    assert np.all(mu <= upper)


def test_mc_propagate_is_deterministic_same_seed() -> None:
    """Two calls with identical inputs + seed produce bit-identical outputs."""
    from pulse_research.coupling.hufner import mc_propagate

    time_mu = np.array([5.0, 6.0, 4.0, 7.0, 3.0])
    time_sigma = np.array([0.2, 0.3, 0.1, 0.4, 0.2])
    o2_mu = np.array([0.95, 0.92, 0.96, 0.90, 0.97])
    o2_sigma = np.array([0.01, 0.015, 0.008, 0.02, 0.005])
    fio2 = np.array([0.21, 0.18, 0.25, 0.15, 0.50])
    out_a = mc_propagate(
        time_mu=time_mu, time_sigma=time_sigma,
        o2_mu=o2_mu, o2_sigma=o2_sigma,
        fio2=fio2, n_mc=512, seed=42, alpha=0.10,
    )
    out_b = mc_propagate(
        time_mu=time_mu, time_sigma=time_sigma,
        o2_mu=o2_mu, o2_sigma=o2_sigma,
        fio2=fio2, n_mc=512, seed=42, alpha=0.10,
    )
    for a, b in zip(out_a, out_b, strict=True):
        np.testing.assert_array_equal(a, b)
