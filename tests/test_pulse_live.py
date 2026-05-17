# tests/test_pulse_live.py
"""Live integration smoke for the Pulse Engine docker subprocess path.

Gated by PULSE_LIVE=1. CI never sets this; maintainers opt in locally.
Requires pulse-ds:4.3.1 image to be built first
(see /root/repos/SKILLS.md/pulse-sim/INSTALL.md and
docs/research/phase7_0_5_integration_smoke.md).

Contract notes (locked by deep_dive_1 §2 and pulse_glue.py):
  - Pulse is the hypoxia orthogonal oracle; it models FiO2/SaO2 trajectories.
  - ``time_to_gloc_s`` is always None from Pulse (G-LOC is CGEM's domain).
  - ``cerebral_o2_min`` is the live Pulse signal (min SpO2 fraction, 0-1).
  - Failure is signalled by error is not None or cerebral_o2_min is None.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pulse_research.api.pulse_glue import make_pulse_row_fn


def _skip_if_not_live() -> None:
    if os.environ.get("PULSE_LIVE") != "1":
        pytest.skip("set PULSE_LIVE=1 to run the live Pulse docker smoke")


def _baseline_row(*, fio2: float) -> np.ndarray:
    """11-axis design row matching AXIS_NAMES order."""
    return np.array(
        [3.0,    # gz_peak — low so we don't immediately hit G-LOC
         1.0,    # gz_onset_rate
         15.0,   # seat_tilt_deg
         0.5,    # anti_g_strain
         80.0,   # pilot_weight_kg
         175.0,  # pilot_height_cm
         35.0,   # pilot_age_y
         50.0,   # baseline_vo2max
         90.0,   # baseline_map_mmhg
         fio2,
         0.97],  # sao2_baseline
        dtype=float,
    )


def test_pulse_live_normobaric_row() -> None:
    _skip_if_not_live()
    row_fn = make_pulse_row_fn()
    output = row_fn(_baseline_row(fio2=0.21))
    assert output.error is None
    # Pulse does not model G-LOC events; time_to_gloc_s is always None
    assert output.time_to_gloc_s is None
    assert output.cerebral_o2_min is not None
    assert 0.0 < output.cerebral_o2_min < 1.0


def test_pulse_live_hypoxic_row() -> None:
    _skip_if_not_live()
    row_fn = make_pulse_row_fn()
    output = row_fn(_baseline_row(fio2=0.15))
    assert output.error is None
    # Pulse does not model G-LOC events; time_to_gloc_s is always None
    assert output.time_to_gloc_s is None
    assert output.cerebral_o2_min is not None
    assert 0.0 < output.cerebral_o2_min < 1.0
