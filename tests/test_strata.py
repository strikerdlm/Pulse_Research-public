"""Tests for the FiO2 tier Mondrian-strata binning."""
from __future__ import annotations

import numpy as np
import pytest


def test_boundary_values() -> None:
    from pulse_research.sensitivity.strata import fio2_tier

    fio2 = np.array([1.0, 0.21, 0.20, 0.199, 0.16, 0.159, 0.15])
    tier = fio2_tier(fio2)
    expected = np.array(
        ["normo", "normo", "normo", "hypo3000", "hypo3000", "hypo4000", "hypo4000"]
    )
    np.testing.assert_array_equal(tier, expected)


def test_vectorized_over_2d_input_raises() -> None:
    from pulse_research.sensitivity.strata import fio2_tier

    with pytest.raises(ValueError, match="1-D"):
        fio2_tier(np.array([[0.21], [0.18]]))


def test_dtype_is_string() -> None:
    from pulse_research.sensitivity.strata import fio2_tier

    out = fio2_tier(np.array([0.21, 0.18, 0.14]))
    assert out.dtype.kind in ("U", "O")
