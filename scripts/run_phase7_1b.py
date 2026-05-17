#!/usr/bin/env python3
"""Phase 7.1b production paired run.

Drives ``make_cgem_surrogate_row_fn`` + ``make_pulse_row_fn`` per row at
``n_base=128`` (3 072 design rows per arm) and writes:

* ``<out_dir>/run_records_phase7_1b.parquet`` — 6 144 RunRecords.
* ``<out_dir>/run_records_phase7_1b.manifest.json`` — provenance.

No fabricated data: every populated ``time_to_gloc_s`` is a live call
to ``surrogate.predict_expected_time_array``; every populated
``cerebral_o2_min`` is a live Pulse Docker subprocess return. Errors
are recorded as RunRecords with both output fields ``None`` —
downstream auditing and pairing depend on this contract.

Pre-flight checks run before any compute:

* ``docker inspect`` confirms the Pulse image is present locally; exit 1
  if not.
* CGEM training parquet must be readable and is fingerprinted via
  SHA-256 before the surrogate is fit; exit 1 on read failure.

Usage::

    python scripts/run_phase7_1b.py \\
        [--n-base 128] [--seed 42] [--out-dir data/] \\
        [--pulse-image pulse-ds:4.3.1]
"""
from __future__ import annotations

import argparse
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from pulse_research.api.cgem_glue import make_cgem_surrogate_row_fn
from pulse_research.api.pulse_glue import make_pulse_row_fn
from pulse_research.io.records import (
    Fidelity,
    RunRecord,
    write_records_parquet,
)
from pulse_research.provenance.manifest import (
    CgemArmManifest,
    DesignManifest,
    ProvenanceManifest,
    PulseArmManifest,
    compute_parquet_sha256,
    write_manifest,
)
from pulse_research.schema import AntiGSuit, FeatureVector19
from pulse_research.sensitivity.sobol_design import AXIS_NAMES, build_design

CGEM_PARQUET = Path(
    "/root/repos/CAMI-Gz-Effects-Model-CGEM-/data/datasets/cgem_synthetic_v1.parquet"
)


@dataclass
class _ArmTiming:
    started_at: str
    finished_at: str
    wall_clock_s: float


def _row_to_features(row: np.ndarray) -> FeatureVector19:
    """Build a FeatureVector19 from an 11-axis design row.

    Fixed-per-batch covariates (sex, hypocapnia, suit class, etc.) are
    held at baseline values; these are NOT part of the Saltelli design.
    """
    return FeatureVector19(
        gz_peak=float(row[0]),
        gz_onset_rate=float(row[1]),
        seat_tilt_deg=float(row[2]),
        anti_g_strain=float(row[3]),
        pilot_weight_kg=float(row[4]),
        pilot_height_cm=float(row[5]),
        pilot_age_y=float(row[6]),
        baseline_vo2max=float(row[7]),
        baseline_map_mmhg=float(row[8]),
        fio2_inspired=float(row[9]),
        sao2_baseline=float(row[10]),
        sex_male=True,
        hypocapnia_flag=False,
        anti_g_suit_class=AntiGSuit.ATAGS,
        retinal_baseline_perfusion=0.85,
        cerebral_autoreg_gain=1.0,
        pulmonary_shunt_baseline=0.02,
        baseline_hb_g_dl=14.5,
        ambient_temp_c=22.0,
    )


def _docker_inspect_digest(image: str) -> str:
    """Return the local Docker image ID; exit 1 if the image is missing."""
    res = subprocess.run(
        ["docker", "inspect", "--format={{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        sys.stderr.write(
            f"ERROR: Docker image {image!r} not present locally. "
            f"Build it first via the Phase 7.0.5 instructions. "
            f"stderr: {res.stderr.strip()}\n"
        )
        sys.exit(1)
    return res.stdout.strip()


def _xgboost_version() -> str:
    try:
        import xgboost

        return str(xgboost.__version__)
    except ImportError:
        return "unavailable"


def _runtime_info() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "xgboost_version": _xgboost_version(),
        "hostname": socket.gethostname(),
    }


def _run_cgem_arm(
    design: np.ndarray,
    training_parquet: Path,
) -> tuple[list[RunRecord], _ArmTiming, dict[str, object]]:
    print(f"Training CGEM surrogate from {training_parquet}...")
    t0 = time.time()
    row_fn = make_cgem_surrogate_row_fn(training_parquet)
    print(f"  -> trained in {time.time() - t0:.2f}s")

    started_at = datetime.now().isoformat()
    t0 = time.time()
    records: list[RunRecord] = []
    for i in range(design.shape[0]):
        out = row_fn(design[i])
        records.append(
            RunRecord(
                run_id=f"phase7_1b-cgem-{i:04d}",
                fidelity=Fidelity.LOW,
                features=_row_to_features(design[i]),
                time_to_gloc_s=out.time_to_gloc_s,
                cerebral_o2_min=None,
                engine_version="cgem-surrogate-v1",
            )
        )
    wall = time.time() - t0
    finished_at = datetime.now().isoformat()
    print(f"  -> {len(records)} CGEM records in {wall:.2f}s")
    # The make_cgem_surrogate_row_fn closure does not expose fit_info; we
    # re-train once more in a no-op path to capture it without growing the
    # row_fn surface area. The training cost is ~0.4 s on the synthetic
    # dataset and the result is byte-equivalent to the row_fn's surrogate.
    fit_info = _capture_fit_info(training_parquet)
    return records, _ArmTiming(started_at, finished_at, wall), fit_info


def _capture_fit_info(training_parquet: Path) -> dict[str, object]:
    """Re-train the surrogate just to expose its fit_info for the manifest.

    The closure built by ``make_cgem_surrogate_row_fn`` does not return the
    surrogate object; we re-fit here for provenance only. The XGBoost
    training is deterministic given the parquet, so the two fits produce
    identical models — this only costs the ~0.4 s training overhead.
    """
    import pandas as pd
    from cgem_ext.surrogate.xgb import build_surrogate

    df = pd.read_parquet(training_parquet)
    df = df[df["status"] == "ok"].copy()
    surrogate = build_surrogate("time_to_gloc_s").fit(df)
    info = surrogate.fit_info
    return {
        "target": info.target,
        "censored": info.censored,
        "n_train": info.n_train,
        "n_train_event": info.n_train_event,
        "feature_columns": list(info.feature_columns),
    }


def _run_pulse_arm(
    design: np.ndarray,
    pulse_image: str,
) -> tuple[list[RunRecord], _ArmTiming, int]:
    print(f"Running Pulse arm via {pulse_image}...")
    started_at = datetime.now().isoformat()
    t0 = time.time()
    row_fn = make_pulse_row_fn()
    records: list[RunRecord] = []
    error_count = 0
    n_rows = design.shape[0]
    for i in range(n_rows):
        out = row_fn(design[i])
        features = _row_to_features(design[i])
        if out.error is not None or out.cerebral_o2_min is None:
            records.append(
                RunRecord(
                    run_id=f"phase7_1b-pulse-{i:04d}",
                    fidelity=Fidelity.HIGH,
                    features=features,
                    time_to_gloc_s=None,
                    cerebral_o2_min=None,
                    engine_version="pulse-4.3.1",
                )
            )
            error_count += 1
            continue
        records.append(
            RunRecord(
                run_id=f"phase7_1b-pulse-{i:04d}",
                fidelity=Fidelity.HIGH,
                features=features,
                time_to_gloc_s=None,
                cerebral_o2_min=out.cerebral_o2_min,
                engine_version="pulse-4.3.1",
            )
        )
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_rows - i - 1) / rate if rate > 0 else float("inf")
            print(
                f"  pulse {i + 1}/{n_rows} | rate {rate:.2f} rows/s | "
                f"errors {error_count} | ETA {eta / 60:.1f} min"
            )
    wall = time.time() - t0
    finished_at = datetime.now().isoformat()
    print(
        f"  -> {len(records)} Pulse records in {wall / 60:.2f} min "
        f"({error_count} errors)"
    )
    return records, _ArmTiming(started_at, finished_at, wall), error_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-base", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data/"))
    ap.add_argument("--pulse-image", type=str, default="pulse-ds:4.3.1")
    args = ap.parse_args()

    # Pre-flight: Docker image must be present.
    image_digest = _docker_inspect_digest(args.pulse_image)
    print(f"Pulse image: {args.pulse_image} (digest {image_digest})")

    # Pre-flight: training parquet readable + SHA-256 captured.
    if not CGEM_PARQUET.is_file():
        sys.stderr.write(
            f"ERROR: CGEM training parquet not found at {CGEM_PARQUET}.\n"
        )
        sys.exit(1)
    print(f"Hashing training parquet at {CGEM_PARQUET}...")
    t0 = time.time()
    training_sha = compute_parquet_sha256(CGEM_PARQUET)
    print(f"  -> SHA-256 {training_sha} in {time.time() - t0:.2f}s")

    # Build the Saltelli design.
    design, axis_names = build_design(n_base=args.n_base, seed=args.seed)
    assert axis_names == AXIS_NAMES
    n_rows = design.shape[0]
    print(f"Saltelli design: n_base={args.n_base}, seed={args.seed}, n_rows={n_rows}")

    # Run both arms.
    cgem_records, cgem_timing, cgem_fit_info = _run_cgem_arm(design, CGEM_PARQUET)
    pulse_records, pulse_timing, pulse_errors = _run_pulse_arm(
        design, args.pulse_image
    )

    # Write parquet.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out_dir / "run_records_phase7_1b.parquet"
    write_records_parquet([*cgem_records, *pulse_records], parquet_path)
    print(f"Wrote {len(cgem_records) + len(pulse_records)} records to {parquet_path}")

    # Build + write manifest.
    manifest = ProvenanceManifest(
        phase="7.1b",
        seed=args.seed,
        n_pulse_base=args.n_base,
        n_cgem_base=args.n_base,
        runtime=_runtime_info(),
        design=DesignManifest(
            axis_names=list(axis_names),
            n_rows=n_rows,
            saltelli_calc_second_order=True,
        ),
        cgem=CgemArmManifest(
            training_parquet_path=str(CGEM_PARQUET),
            training_parquet_sha256=training_sha,
            fit_info=cgem_fit_info,
            output_channel="predict_expected_time_array",
            row_count=len(cgem_records),
            error_count=0,
            wall_clock_s=cgem_timing.wall_clock_s,
            started_at=cgem_timing.started_at,
            finished_at=cgem_timing.finished_at,
        ),
        pulse=PulseArmManifest(
            docker_image=args.pulse_image,
            docker_image_digest=image_digest,
            row_count=len(pulse_records),
            error_count=pulse_errors,
            wall_clock_s=pulse_timing.wall_clock_s,
            started_at=pulse_timing.started_at,
            finished_at=pulse_timing.finished_at,
        ),
    )
    manifest_path = args.out_dir / "run_records_phase7_1b.manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"Wrote manifest to {manifest_path}")

    _print_summary(manifest, cgem_records, pulse_records)


def _print_summary(
    manifest: ProvenanceManifest,
    cgem_records: list[RunRecord],
    pulse_records: list[RunRecord],
) -> None:
    cgem_times = [
        r.time_to_gloc_s for r in cgem_records if r.time_to_gloc_s is not None
    ]
    pulse_o2 = [
        r.cerebral_o2_min for r in pulse_records if r.cerebral_o2_min is not None
    ]
    print("\n=== Phase 7.1b run summary ===")
    print(
        f"CGEM:  {manifest.cgem.row_count} rows | "
        f"wall {manifest.cgem.wall_clock_s:.2f}s"
    )
    if cgem_times:
        print(
            f"  time_to_gloc_s P*E: min={min(cgem_times):.4f} "
            f"max={max(cgem_times):.4f} mean={np.mean(cgem_times):.4f}"
        )
    print(
        f"Pulse: {manifest.pulse.row_count} rows | "
        f"errors {manifest.pulse.error_count} | "
        f"wall {manifest.pulse.wall_clock_s / 60:.2f} min"
    )
    if pulse_o2:
        print(
            f"  cerebral_o2_min: min={min(pulse_o2):.4f} "
            f"max={max(pulse_o2):.4f} mean={np.mean(pulse_o2):.4f}"
        )


if __name__ == "__main__":
    main()
