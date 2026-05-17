"""Tests for the CGEM glue.

No real CGEM Fortran binary is executed; every test either uses the pure
mapping function or injects a fake ``cgem_module`` so the test stays hermetic.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pulse_research.api.cgem_glue import (
    CGEM_ENV_VAR,
    design_row_to_centrifuge_params,
    make_cgem_row_fn,
)
from pulse_research.api.runners import RowOutput


@dataclass(frozen=True)
class _FakePilotConfig:
    """Mirror the subset of upstream ``cgem_wrapper.PilotConfig`` we touch."""

    who_profile: int | None = 2
    height_cm: float | None = 179.0
    seat_tilt_deg: float = 10.0
    agsm_effectiveness: float = 0.0
    baseline_systolic_bp: float | None = 120.0
    baseline_diastolic_bp: float | None = 80.0


def _baseline_row() -> np.ndarray:
    """A reasonable 11-axis Sobol row."""
    return np.array(
        [6.0, 3.0, 15.0, 0.5, 80.0, 175.0, 35.0, 50.0, 90.0, 0.21, 0.97],
        dtype=float,
    )


def test_design_row_to_centrifuge_params_locks_mapping() -> None:
    row = _baseline_row()
    params = design_row_to_centrifuge_params(row, PilotConfigCls=_FakePilotConfig)
    # Direct mappings
    assert params["g0"] == 1.0
    assert params["gmax"] == 6.0
    assert params["gmaxtime"] == 15.0
    # rampup = (gmax - g0) / onset_rate = (6 - 1) / 3 = 1.6667
    assert params["rampup"] == pytest.approx(5.0 / 3.0, rel=1e-9)
    assert params["rampdown"] == params["rampup"]
    cfg = params["config"]
    assert cfg.who_profile is None
    assert cfg.height_cm == 175.0
    assert cfg.seat_tilt_deg == 15.0
    assert cfg.agsm_effectiveness == 0.5
    assert cfg.baseline_systolic_bp == 110.0  # 90 + 20
    assert cfg.baseline_diastolic_bp == 80.0  # 90 - 10


def test_design_row_to_centrifuge_params_clips_agsm() -> None:
    row = _baseline_row()
    row[3] = 1.5  # out-of-range strain
    params = design_row_to_centrifuge_params(row, PilotConfigCls=_FakePilotConfig)
    assert params["config"].agsm_effectiveness == 1.0


def test_design_row_to_centrifuge_params_floor_on_rampup() -> None:
    row = _baseline_row()
    row[0] = 1.2  # gmax very close to g0=1.0
    row[1] = 6.0  # high onset rate
    params = design_row_to_centrifuge_params(row, PilotConfigCls=_FakePilotConfig)
    assert params["rampup"] >= 0.1


def test_design_row_to_centrifuge_params_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        design_row_to_centrifuge_params(
            np.zeros((5,)), PilotConfigCls=_FakePilotConfig
        )


def _fake_cgem_module(
    *, return_gloc: float | None = 11.5, raises: type[BaseException] | None = None
) -> Any:
    """Build a stub cgem_wrapper module with the surface we use."""

    def run_cgem_centrifuge(**_kwargs: Any) -> tuple[Any, Path]:
        if raises is not None:
            raise raises("simulated CGEM failure")
        return (SimpleNamespace(time_to_gloc_s=return_gloc), Path("/tmp/fake"))

    return SimpleNamespace(
        PilotConfig=_FakePilotConfig,
        run_cgem_centrifuge=run_cgem_centrifuge,
    )


def test_make_cgem_row_fn_returns_rowoutput_on_success() -> None:
    row_fn = make_cgem_row_fn(_fake_cgem_module(return_gloc=11.5))
    out = row_fn(_baseline_row())
    assert isinstance(out, RowOutput)
    assert out.time_to_gloc_s == 11.5
    assert out.cerebral_o2_min is None
    assert out.error is None


def test_make_cgem_row_fn_passes_through_none_time_to_gloc() -> None:
    """CGEM may return None when the profile never reaches G-LOC; preserve it."""
    row_fn = make_cgem_row_fn(_fake_cgem_module(return_gloc=None))
    out = row_fn(_baseline_row())
    assert out.time_to_gloc_s is None
    assert out.error is None


def test_make_cgem_row_fn_swallows_subprocess_exception() -> None:
    row_fn = make_cgem_row_fn(_fake_cgem_module(raises=RuntimeError))
    out = row_fn(_baseline_row())
    assert out.time_to_gloc_s is None
    assert out.cerebral_o2_min is None
    assert out.error is not None
    assert "CGEM failure" in out.error


def test_make_cgem_row_fn_missing_env_var_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CGEM_ENV_VAR, raising=False)
    if CGEM_ENV_VAR in os.environ:
        pytest.skip("env var sticky in this process — skipping")
    with pytest.raises(RuntimeError, match="CGEM_ROOT"):
        make_cgem_row_fn()
