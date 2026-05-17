#!/usr/bin/env python3
"""Phase 7.1a small-N paired smoke.

Drives make_cgem_surrogate_row_fn + make_pulse_row_fn per row at n_base=8
(192 design rows) and writes a paired RunRecord parquet. CGEM-surrogate
arm runs unconditionally (no Docker needed); Pulse arm runs only when
PULSE_LIVE=1 is set in the environment.

Usage::

    python scripts/run_paired_smoke.py [--n-base 8] [--seed 42] [--out PATH]

Default output: data/run_records_smoke.parquet
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from pulse_research.api.cgem_glue import make_cgem_surrogate_row_fn
from pulse_research.api.pulse_glue import make_pulse_row_fn
from pulse_research.io.records import (
    Fidelity,
    RunRecord,
    write_records_parquet,
)
from pulse_research.schema import AntiGSuit, FeatureVector19
from pulse_research.sensitivity.sobol_design import AXIS_NAMES, build_design

CGEM_PARQUET = Path(
    "/root/repos/CAMI-Gz-Effects-Model-CGEM-/data/datasets/cgem_synthetic_v1.parquet"
)


def _row_to_features(row: np.ndarray) -> FeatureVector19:
    """Build a FeatureVector19 from an 11-axis design row.

    Fixed-per-batch covariates (sex, hypocapnia, suit class, etc.) are
    held at baseline values; these match the smoke-script convention
    and are NOT part of the Saltelli design.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-base", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/run_records_smoke.parquet"),
        help="Output parquet path (default: data/run_records_smoke.parquet)",
    )
    args = ap.parse_args()

    pulse_live = os.environ.get("PULSE_LIVE") == "1"

    print(f"Saltelli design: n_base={args.n_base}, seed={args.seed}")
    design, names = build_design(n_base=args.n_base, seed=args.seed)
    assert names == AXIS_NAMES
    n_rows = design.shape[0]
    print(f"  -> {n_rows} design rows x {design.shape[1]} axes")

    print(f"Training CGEM surrogate from {CGEM_PARQUET}...")
    t0 = time.time()
    cgem_row_fn = make_cgem_surrogate_row_fn(CGEM_PARQUET)
    print(f"  -> trained in {time.time() - t0:.2f}s")

    records: list[RunRecord] = []

    print("Running CGEM-surrogate arm...")
    t0 = time.time()
    for i in range(n_rows):
        out = cgem_row_fn(design[i])
        features = _row_to_features(design[i])
        records.append(
            RunRecord(
                run_id=f"smoke-cgem-{i:04d}",
                fidelity=Fidelity.LOW,
                features=features,
                time_to_gloc_s=out.time_to_gloc_s,
                cerebral_o2_min=out.cerebral_o2_min,
                engine_version="cgem-surrogate-v1",
            )
        )
    print(f"  -> {n_rows} CGEM records in {time.time() - t0:.2f}s")

    if pulse_live:
        print("Running Pulse arm (PULSE_LIVE=1)...")
        t0 = time.time()
        pulse_row_fn = make_pulse_row_fn()
        for i in range(n_rows):
            out = pulse_row_fn(design[i])
            features = _row_to_features(design[i])
            records.append(
                RunRecord(
                    run_id=f"smoke-pulse-{i:04d}",
                    fidelity=Fidelity.HIGH,
                    features=features,
                    time_to_gloc_s=out.time_to_gloc_s,
                    cerebral_o2_min=out.cerebral_o2_min,
                    engine_version="pulse-4.3.1",
                )
            )
        print(f"  -> {n_rows} Pulse records in {time.time() - t0:.2f}s")
    else:
        print("PULSE_LIVE unset; skipping Pulse arm.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_records_parquet(records, args.out)
    print(f"Wrote {len(records)} records to {args.out}")

    cgem_times = [
        r.time_to_gloc_s
        for r in records
        if r.fidelity == Fidelity.LOW and r.time_to_gloc_s is not None
    ]
    if cgem_times:
        print(
            f"CGEM time_to_gloc_s: min={min(cgem_times):.2f}, "
            f"max={max(cgem_times):.2f}, mean={np.mean(cgem_times):.2f}"
        )
    pulse_o2 = [
        r.cerebral_o2_min
        for r in records
        if r.fidelity == Fidelity.HIGH and r.cerebral_o2_min is not None
    ]
    if pulse_o2:
        print(
            f"Pulse cerebral_o2_min: min={min(pulse_o2):.4f}, "
            f"max={max(pulse_o2):.4f}, mean={np.mean(pulse_o2):.4f}"
        )


if __name__ == "__main__":
    main()
