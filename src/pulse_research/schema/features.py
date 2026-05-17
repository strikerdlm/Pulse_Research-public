"""19-d feature schema for the multi-fidelity CGEM-Pulse surrogate.

The vector is the 11 Sobol-design axes
(see :mod:`pulse_research.sensitivity.sobol_design`) plus 8 fixed covariates
that stay constant within a Sobol batch but vary across cohorts:

* sex_male (bool)
* hypocapnia_flag (bool)
* anti_g_suit_class (enum)
* retinal_baseline_perfusion (0.5 to 1.0, fraction)
* cerebral_autoreg_gain (0.5 to 1.5, dimensionless)
* pulmonary_shunt_baseline (0 to 0.3, fraction)
* baseline_hb_g_dl (10 to 18 g/dL)
* ambient_temp_c (-20 to 45 C)

Bounds on the 11 Sobol axes here MUST stay in sync with
:data:`pulse_research.sensitivity.sobol_design.AXIS_BOUNDS`; a unit test
checks the invariant.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AntiGSuit(StrEnum):
    """Anti-G suit classes covered by the surrogate's pilot covariate."""

    NONE = "NONE"
    CSU_15P = "CSU_15P"
    COMBAT_EDGE = "COMBAT_EDGE"
    ATAGS = "ATAGS"
    EAGLE_FIT = "EAGLE_FIT"


class FeatureVector19(BaseModel):
    """Immutable 19-d input feature for the surrogate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gz_peak: float = Field(..., ge=1.0, le=9.0)
    gz_onset_rate: float = Field(..., ge=0.1, le=6.0)
    seat_tilt_deg: float = Field(..., ge=0.0, le=30.0)
    anti_g_strain: float = Field(..., ge=0.0, le=1.0)
    pilot_weight_kg: float = Field(..., ge=50.0, le=110.0)
    pilot_height_cm: float = Field(..., ge=155.0, le=200.0)
    pilot_age_y: float = Field(..., ge=20.0, le=55.0)
    baseline_vo2max: float = Field(..., ge=30.0, le=70.0)
    baseline_map_mmhg: float = Field(..., ge=70.0, le=110.0)
    fio2_inspired: float = Field(..., ge=0.15, le=1.00)
    sao2_baseline: float = Field(..., ge=0.80, le=1.00)

    sex_male: bool
    hypocapnia_flag: bool
    anti_g_suit_class: AntiGSuit
    retinal_baseline_perfusion: float = Field(..., ge=0.5, le=1.0)
    cerebral_autoreg_gain: float = Field(..., ge=0.5, le=1.5)
    pulmonary_shunt_baseline: float = Field(..., ge=0.0, le=0.3)
    baseline_hb_g_dl: float = Field(..., ge=10.0, le=18.0)
    ambient_temp_c: float = Field(..., ge=-20.0, le=45.0)
