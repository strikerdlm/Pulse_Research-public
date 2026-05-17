"""Tests for the Pulse glue.

No Docker is invoked: every test either calls a pure function or injects a
fake ``subprocess_runner`` and ``work_dir_factory`` into the row-fn builder.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pulse_research.api.pulse_glue import (
    DEFAULT_PULSE_IMAGE,
    PULSE_IMAGE_ENV_VAR,
    design_row_to_scenario_params,
    extract_pulse_outputs,
    make_pulse_row_fn,
    render_pulse_script,
)
from pulse_research.api.runners import RowOutput


def _baseline_row(fio2: float = 0.21) -> np.ndarray:
    row = np.array(
        [6.0, 3.0, 15.0, 0.5, 80.0, 175.0, 35.0, 50.0, 90.0, fio2, 0.97],
        dtype=float,
    )
    return row


def test_design_row_to_scenario_params_fio2_high_picks_standard() -> None:
    params = design_row_to_scenario_params(_baseline_row(fio2=0.21))
    assert params["environment_file"].endswith("Standard.json")
    assert params["fio2_inspired"] == pytest.approx(0.21)


def test_design_row_to_scenario_params_fio2_mid_picks_hypobaric_3000m() -> None:
    params = design_row_to_scenario_params(_baseline_row(fio2=0.17))
    assert params["environment_file"].endswith("Hypobaric3000m.json")


def test_design_row_to_scenario_params_fio2_low_picks_hypobaric_4000m() -> None:
    params = design_row_to_scenario_params(_baseline_row(fio2=0.15))
    assert params["environment_file"].endswith("Hypobaric4000m.json")


def test_design_row_to_scenario_params_passes_duration() -> None:
    params = design_row_to_scenario_params(_baseline_row(), duration_s=42.0)
    assert params["duration_s"] == 42.0


def test_design_row_to_scenario_params_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        design_row_to_scenario_params(np.zeros((5,)))


def test_render_pulse_script_includes_environment_and_duration() -> None:
    params = design_row_to_scenario_params(_baseline_row(fio2=0.15), duration_s=42.0)
    script = render_pulse_script(params)
    assert "Hypobaric4000m.json" in script
    assert "advance_time_s(42.0)" in script
    assert "OxygenSaturation" in script  # data request still present


def test_extract_pulse_outputs_finds_oxygen_saturation_min() -> None:
    df = pd.DataFrame(
        {
            "Time(s)": [0.0, 30.0, 60.0],
            "OxygenSaturation": [0.99, 0.87, 0.85],
            "HeartRate(1/min)": [72.0, 90.0, 115.0],
        }
    )
    out = extract_pulse_outputs(df)
    assert out["cerebral_o2_min"] == pytest.approx(0.85)
    assert out["time_to_gloc_s"] is None


def test_extract_pulse_outputs_missing_column_returns_none() -> None:
    df = pd.DataFrame({"Time(s)": [0.0, 1.0], "HeartRate(1/min)": [72.0, 73.0]})
    out = extract_pulse_outputs(df)
    assert out["cerebral_o2_min"] is None
    assert out["time_to_gloc_s"] is None


def _fake_subprocess_runner_factory(
    write_csv: pd.DataFrame | None = None,
    raises: type[BaseException] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build a fake subprocess.run that records calls and optionally writes
    a CSV to where the runner expects it before returning."""
    calls: list[dict[str, Any]] = []

    def fake(cmd, *, check, timeout, capture_output):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "cmd": list(cmd),
                "check": check,
                "timeout": timeout,
                "capture_output": capture_output,
            }
        )
        if raises is not None:
            raise raises("fake docker failure")
        # The 5th and 7th elements of the cmd are the bind-mount strings.
        results_bind = next(
            arg for arg in cmd if isinstance(arg, str) and arg.endswith(":/pulse/bin/test_results")
        )
        host_results_dir = Path(results_bind.split(":", 1)[0])
        if write_csv is not None:
            write_csv.to_csv(host_results_dir / "scenario.csv", index=False)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    return fake, calls


def test_make_pulse_row_fn_with_mocked_subprocess_success(tmp_path: Path) -> None:
    csv_df = pd.DataFrame(
        {
            "Time(s)": [0.0, 30.0, 60.0],
            "OxygenSaturation": [0.98, 0.88, 0.82],
        }
    )
    fake_run, calls = _fake_subprocess_runner_factory(write_csv=csv_df)

    work_dirs: list[Path] = []

    def factory() -> Path:
        d = tmp_path / f"row-{len(work_dirs)}"
        d.mkdir()
        work_dirs.append(d)
        return d

    row_fn = make_pulse_row_fn(
        image="pulse-ds:test",
        work_dir_factory=factory,
        subprocess_runner=fake_run,
    )

    out = row_fn(_baseline_row(fio2=0.15))
    assert isinstance(out, RowOutput)
    assert out.error is None
    assert out.cerebral_o2_min == pytest.approx(0.82)
    assert out.time_to_gloc_s is None
    assert len(calls) == 1
    assert calls[0]["cmd"][0] == "docker"
    assert "pulse-ds:test" in calls[0]["cmd"]
    # The rendered script was written and is on the bind mount
    rendered = (work_dirs[0] / "scenario.py").read_text()
    assert "Hypobaric4000m.json" in rendered


def test_make_pulse_row_fn_subprocess_failure_yields_error_rowoutput(
    tmp_path: Path,
) -> None:
    fake_run, _ = _fake_subprocess_runner_factory(raises=RuntimeError)
    row_fn = make_pulse_row_fn(
        image="pulse-ds:test",
        work_dir_factory=lambda: tmp_path,
        subprocess_runner=fake_run,
    )
    out = row_fn(_baseline_row())
    assert out.time_to_gloc_s is None
    assert out.cerebral_o2_min is None
    assert out.error is not None
    assert "fake docker failure" in out.error


def test_make_pulse_row_fn_picks_up_env_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PULSE_IMAGE_ENV_VAR, "pulse-ds:override")
    csv_df = pd.DataFrame(
        {"Time(s)": [0.0], "OxygenSaturation": [0.95]}
    )
    fake_run, calls = _fake_subprocess_runner_factory(write_csv=csv_df)
    row_fn = make_pulse_row_fn(
        work_dir_factory=lambda: tmp_path,
        subprocess_runner=fake_run,
    )
    row_fn(_baseline_row())
    assert "pulse-ds:override" in calls[0]["cmd"]


def test_default_pulse_image_constant() -> None:
    assert DEFAULT_PULSE_IMAGE == "pulse-ds:4.3.1"
