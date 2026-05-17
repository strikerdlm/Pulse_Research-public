"""Tests for the CGEM surrogate integration (Phase 7.1a)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pulse_research.api.cgem_glue import (
    design_row_to_surrogate_features,
    make_cgem_surrogate_row_fn,
)

_CGEM_PARQUET = Path(
    "/root/repos/CAMI-Gz-Effects-Model-CGEM-/data/datasets/cgem_synthetic_v1.parquet"
)


def _skip_if_no_cgem_data() -> None:
    try:
        present = _CGEM_PARQUET.is_file()
    except (PermissionError, OSError):
        present = False
    if not present:
        pytest.skip(
            f"CGEM training data not accessible at {_CGEM_PARQUET}; "
            "CGEM_ROOT may be unset, the dataset may not be built, or "
            "the path may be unreadable on this runner"
        )


def _design_center_row() -> np.ndarray:
    """11-axis row at the design center where g_tolerance_multiplier == 1.0."""
    return np.array(
        [3.0,    # gz_peak
         1.0,    # gz_onset_rate
         0.0,    # seat_tilt_deg (no tilt contribution at center)
         0.5,    # anti_g_strain
         80.0,   # pilot_weight_kg
         175.0,  # pilot_height_cm
         30.0,   # pilot_age_y (no age contribution at center)
         50.0,   # baseline_vo2max (no fitness contribution at center)
         90.0,   # baseline_map_mmhg (no MAP contribution at center)
         0.21,   # fio2_inspired (ignored by CGEM)
         0.97],  # sao2_baseline (ignored by CGEM)
        dtype=float,
    )


def test_translator_maps_design_row_to_17d_dataframe() -> None:
    df = design_row_to_surrogate_features(_design_center_row())
    expected_columns = {
        "g_peak_abs", "dgdt_max_g_per_s", "profile_duration_s",
        "dehydration_level", "g_tolerance_multiplier",
        "gsuit_max_psi", "gsuit_coverage_fraction", "agsm_effectiveness",
        "pbg_max_mmhg",
        "who_1", "who_2", "who_3", "who_4", "who_5", "who_6", "who_custom",
        "cm_ordinal",
    }
    assert set(df.columns) == expected_columns
    assert len(df) == 1
    assert df["g_tolerance_multiplier"].iloc[0] == pytest.approx(1.0)
    assert df["who_custom"].iloc[0] == 1.0


def test_translator_g_tolerance_multiplier_clip() -> None:
    # Extreme high corner: MAP=200, vo2max=100, age=20, tilt=30
    # Raw = 1.0 + 0.005*(200-90) + 0.010*(100-50) - 0.005*(20-30) + 0.005*30
    #     = 1.0 + 0.55 + 0.50 + 0.05 + 0.15 = 2.25 -> clipped to 1.15
    row = np.array(
        [3.0, 1.0, 30.0, 0.5, 80.0, 175.0, 20.0, 100.0, 200.0, 0.21, 0.97],
        dtype=float,
    )
    df = design_row_to_surrogate_features(row)
    assert df["g_tolerance_multiplier"].iloc[0] == pytest.approx(1.15)


def test_translator_g_tolerance_multiplier_clips_low() -> None:
    # Extreme low corner: MAP=70, vo2max=30, age=55, tilt=0
    # Raw = 1.0 + 0.005*(70-90) + 0.010*(30-50) - 0.005*(55-30) + 0.005*0
    #     = 1.0 - 0.10 - 0.20 - 0.125 + 0 = 0.575 -> clipped to 0.85
    row = np.array(
        [3.0, 1.0, 0.0, 0.5, 80.0, 175.0, 55.0, 30.0, 70.0, 0.21, 0.97],
        dtype=float,
    )
    df = design_row_to_surrogate_features(row)
    assert df["g_tolerance_multiplier"].iloc[0] == pytest.approx(0.85)


def test_translator_cm_ordinal_thresholds() -> None:
    for agsm, expected in [(0.05, 0.0), (0.5, 1.0), (0.9, 2.0)]:
        row = np.array(
            [3.0, 1.0, 0.0, agsm, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97],
            dtype=float,
        )
        df = design_row_to_surrogate_features(row)
        assert df["cm_ordinal"].iloc[0] == expected


def test_make_cgem_surrogate_row_fn_returns_rowfn_with_correct_signature() -> None:
    _skip_if_no_cgem_data()
    row_fn = make_cgem_surrogate_row_fn(_CGEM_PARQUET)
    # Use a row at moderate G that the surrogate should handle cleanly.
    row = np.array(
        [6.0, 1.0, 0.0, 0.5, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97],
        dtype=float,
    )
    output = row_fn(row)
    # Orthogonal-oracle: CGEM populates time_to_gloc_s only.
    assert output.error is None
    assert output.time_to_gloc_s is not None
    assert isinstance(output.time_to_gloc_s, float)
    assert output.cerebral_o2_min is None


def test_make_cgem_surrogate_row_fn_predicts_within_sensible_range() -> None:
    _skip_if_no_cgem_data()
    row_fn = make_cgem_surrogate_row_fn(_CGEM_PARQUET)
    row = np.array(
        [6.0, 1.0, 0.0, 0.5, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97],
        dtype=float,
    )
    output = row_fn(row)
    assert output.time_to_gloc_s is not None
    # Conditional event-time E[time|event=1] from predict_array; bounded by
    # the WF2013 LOCINDTI_min floor (5 s) and the training-set max (~16.9 s).
    assert 0.0 <= output.time_to_gloc_s < 600.0


def test_factory_high_gz_yields_short_conditional_time() -> None:
    """Channel-switched 2026-05-16: regressor-only predict_array.

    At high Gz (e.g. 8), G-LOC happens fast: conditional time should be
    near the WF2013 ROR mean of ~9 s (range [5, 18] s in the paper).
    """
    _skip_if_no_cgem_data()
    row_fn = make_cgem_surrogate_row_fn(_CGEM_PARQUET)
    high_g = np.array(
        [8.0, 1.0, 0.0, 0.5, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97],
        dtype=float,
    )
    output = row_fn(high_g)
    assert output.time_to_gloc_s is not None
    # Conditional time floored at 5 s (LOCINDTI_min) and bounded by the
    # training-set regressor range (~16.9 s).
    assert 5.0 <= output.time_to_gloc_s < 20.0


def test_factory_low_gz_outside_event_regime_is_documented() -> None:
    """Channel-switched 2026-05-16: regressor-only predict_array.

    At Gz=2 (below WF2013's 4.7 G threshold), no real G-LOC occurs in
    centrifuge runs. The regressor-only predict_array channel still
    returns a value because it extrapolates from event-positive
    training rows; the value is *physiologically meaningless at this Gz*
    (no event regime). Phase 7.3 documents this and restricts
    manuscript claims to Gz >= 4.7. The test asserts only that the
    function returns a finite number (no NaN, no error).
    """
    _skip_if_no_cgem_data()
    row_fn = make_cgem_surrogate_row_fn(_CGEM_PARQUET)
    low_g = np.array(
        [2.0, 1.0, 0.0, 0.5, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97],
        dtype=float,
    )
    output = row_fn(low_g)
    assert output.time_to_gloc_s is not None
    assert 0.0 < output.time_to_gloc_s < 100.0


def test_factory_conditional_time_decreases_with_gz_peak() -> None:
    """Channel-switched 2026-05-16: regressor-only predict_array.

    In the WF2013 event regime (Gz >= 4.7), higher Gz produces FASTER
    G-LOC and therefore SHORTER conditional LOCINDTI. The regressor
    learned this direction from the training set's event rows. Test
    spans Gz 5-9 to stay within the event regime.
    """
    _skip_if_no_cgem_data()
    row_fn = make_cgem_surrogate_row_fn(_CGEM_PARQUET)
    base = [1.0, 0.0, 0.5, 80.0, 175.0, 30.0, 50.0, 90.0, 0.21, 0.97]
    out_5 = row_fn(np.array([5.0, *base], dtype=float)).time_to_gloc_s
    out_9 = row_fn(np.array([9.0, *base], dtype=float)).time_to_gloc_s
    assert out_5 is not None
    assert out_9 is not None
    # Physiologically: higher Gz -> faster LOC -> shorter LOCINDTI.
    # WF2013 ROR Table 1: 9.20 s @ 9 G vs 11.50 s @ 5 G (decreasing with Gz).
    assert out_9 <= out_5
