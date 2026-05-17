"""Tests for the 11-axis Saltelli-Sobol design generator."""
from __future__ import annotations

import numpy as np
import pytest

from pulse_research.sensitivity.sobol_design import (
    AXIS_BOUNDS,
    AXIS_NAMES,
    build_design,
)


def test_axes_match_deep_dive_1() -> None:
    """Canonical 11-axis order — must not change without updating downstream."""
    assert AXIS_NAMES == [
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
    assert AXIS_BOUNDS["fio2_inspired"] == (0.15, 1.00)
    assert AXIS_BOUNDS["sao2_baseline"] == (0.80, 1.00)


def test_design_shape_default() -> None:
    # N=1024 base, k=11 → N*(2k+2) = 1024*24 = 24576 rows
    X, names = build_design(n_base=1024, seed=42)
    assert X.shape == (24576, 11)
    assert names == AXIS_NAMES


def test_design_bounds_respected() -> None:
    X, _ = build_design(n_base=256, seed=42)
    assert (X[:, 9] >= 0.15).all() and (X[:, 9] <= 1.00).all()
    assert (X[:, 10] >= 0.80).all() and (X[:, 10] <= 1.00).all()
    assert (X[:, 0] >= 1.0).all() and (X[:, 0] <= 9.0).all()
    assert (X[:, 1] >= 0.1).all() and (X[:, 1] <= 6.0).all()


def test_seed_determinism() -> None:
    X1, _ = build_design(n_base=256, seed=42)
    X2, _ = build_design(n_base=256, seed=42)
    np.testing.assert_array_equal(X1, X2)


def test_different_seeds_diverge() -> None:
    X1, _ = build_design(n_base=256, seed=42)
    X2, _ = build_design(n_base=256, seed=43)
    assert not np.array_equal(X1, X2)


def test_n_base_must_be_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        build_design(n_base=1000, seed=42)


def test_axis_bounds_keys_match_names() -> None:
    """AXIS_BOUNDS must contain exactly the same keys as AXIS_NAMES."""
    assert set(AXIS_BOUNDS.keys()) == set(AXIS_NAMES)
    assert len(AXIS_BOUNDS) == 11
