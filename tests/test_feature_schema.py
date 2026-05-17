"""Tests for the 19-d FeatureVector19 pydantic schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from pulse_research.schema.features import AntiGSuit, FeatureVector19
from pulse_research.sensitivity.sobol_design import AXIS_BOUNDS


def _valid_kwargs() -> dict[str, object]:
    return {
        "gz_peak": 6.0,
        "gz_onset_rate": 3.0,
        "seat_tilt_deg": 15.0,
        "anti_g_strain": 0.5,
        "pilot_weight_kg": 80.0,
        "pilot_height_cm": 175.0,
        "pilot_age_y": 35.0,
        "baseline_vo2max": 50.0,
        "baseline_map_mmhg": 90.0,
        "fio2_inspired": 0.21,
        "sao2_baseline": 0.97,
        "sex_male": True,
        "hypocapnia_flag": False,
        "anti_g_suit_class": AntiGSuit.ATAGS,
        "retinal_baseline_perfusion": 0.85,
        "cerebral_autoreg_gain": 1.0,
        "pulmonary_shunt_baseline": 0.02,
        "baseline_hb_g_dl": 14.5,
        "ambient_temp_c": 22.0,
    }


def test_valid_vector_constructs() -> None:
    v = FeatureVector19(**_valid_kwargs())  # type: ignore[arg-type]
    assert v.gz_peak == 6.0
    assert v.fio2_inspired == 0.21
    assert v.anti_g_suit_class is AntiGSuit.ATAGS


def test_out_of_bound_fio2_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["fio2_inspired"] = 1.20
    with pytest.raises(ValidationError):
        FeatureVector19(**kwargs)  # type: ignore[arg-type]


def test_out_of_bound_sao2_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["sao2_baseline"] = 0.50
    with pytest.raises(ValidationError):
        FeatureVector19(**kwargs)  # type: ignore[arg-type]


def test_unknown_suit_rejected() -> None:
    kwargs = _valid_kwargs()
    kwargs["anti_g_suit_class"] = "UNKNOWN_SUIT"
    with pytest.raises(ValidationError):
        FeatureVector19(**kwargs)  # type: ignore[arg-type]


def test_json_roundtrip() -> None:
    payload = _valid_kwargs()
    payload["anti_g_suit_class"] = "ATAGS"
    v = FeatureVector19.model_validate(payload)
    dumped = v.model_dump()
    assert dumped["gz_peak"] == 6.0
    assert dumped["anti_g_suit_class"] == "ATAGS"


def test_schema_has_19_properties() -> None:
    schema = FeatureVector19.model_json_schema()
    assert len(schema["properties"]) == 19


def test_extra_fields_forbidden() -> None:
    kwargs = _valid_kwargs()
    kwargs["unexpected_field"] = "anything"
    with pytest.raises(ValidationError):
        FeatureVector19(**kwargs)  # type: ignore[arg-type]


def test_frozen_instance() -> None:
    v = FeatureVector19(**_valid_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        v.gz_peak = 7.0


def test_first_eleven_bounds_match_sobol_axes() -> None:
    """Schema bounds on the 11 Sobol axes must match AXIS_BOUNDS verbatim."""
    schema = FeatureVector19.model_json_schema()["properties"]
    for name, (lo, hi) in AXIS_BOUNDS.items():
        assert schema[name]["minimum"] == lo, name
        assert schema[name]["maximum"] == hi, name
