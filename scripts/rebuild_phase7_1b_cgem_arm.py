#!/usr/bin/env python3
"""Rebuild only the CGEM arm of the Phase 7.1b parquet (preserves Pulse arm).

After the Phase 7.3 ROR-anticorrelation diagnostic, the CGEM channel was
switched from ``predict_expected_time_array`` (P*E) to ``predict_array``
(regressor-only conditional time) so corrected_time aligns with the
WF2013 LOCINDTI direction in the rapid-onset regime. This rebuild
regenerates the CGEM rows with the new channel and merges them with the
existing Pulse arm (whose Docker subprocess output is unchanged).

Cost: ~10 s (CGEM surrogate training + 3 072 row predictions). The
~5 h Pulse arm is NOT re-run.

Usage::

    python scripts/rebuild_phase7_1b_cgem_arm.py \\
        --parquet data/run_records_phase7_1b.parquet \\
        --manifest data/run_records_phase7_1b.manifest.json
"""
from __future__ import annotations

import argparse
import platform
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pulse_research.api.cgem_glue import make_cgem_surrogate_row_fn
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


def _read_manifest(path: Path) -> ProvenanceManifest:
    return ProvenanceManifest.model_validate_json(path.read_text())


def _row_to_features(row: np.ndarray) -> FeatureVector19:
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


def _capture_fit_info(training_parquet: Path) -> dict[str, object]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-base", type=int, default=128)
    args = ap.parse_args()

    # Read existing manifest to recover provenance + Pulse arm metadata.
    manifest_in = _read_manifest(args.manifest)

    # Re-train the surrogate from the recorded training parquet.
    training_sha = compute_parquet_sha256(CGEM_PARQUET)
    if training_sha != manifest_in.cgem.training_parquet_sha256:
        raise RuntimeError(
            "training parquet SHA-256 mismatch with manifest; refusing to rebuild"
        )
    print(f"Training CGEM surrogate from {CGEM_PARQUET}...")
    t0 = time.time()
    row_fn = make_cgem_surrogate_row_fn(CGEM_PARQUET)
    print(f"  -> trained in {time.time() - t0:.2f}s")

    # Build the Saltelli design (deterministic given seed + n_base).
    design, axis_names = build_design(n_base=args.n_base, seed=args.seed)
    assert axis_names == AXIS_NAMES
    n_rows = design.shape[0]
    print(f"Saltelli design: n_base={args.n_base}, seed={args.seed}, n_rows={n_rows}")

    # Re-generate CGEM rows.
    cgem_started = datetime.now().isoformat()
    t0 = time.time()
    cgem_records: list[RunRecord] = []
    for i in range(n_rows):
        out = row_fn(design[i])
        cgem_records.append(
            RunRecord(
                run_id=f"phase7_1b-cgem-{i:04d}",
                fidelity=Fidelity.LOW,
                features=_row_to_features(design[i]),
                time_to_gloc_s=out.time_to_gloc_s,
                cerebral_o2_min=None,
                engine_version="cgem-surrogate-v1",
            )
        )
    cgem_wall = time.time() - t0
    cgem_finished = datetime.now().isoformat()
    print(f"  -> {len(cgem_records)} CGEM records in {cgem_wall:.2f}s")

    # Load existing Pulse arm rows from the input parquet and rebuild them as
    # RunRecord objects (they're already in the parquet — pull them through
    # unchanged so determinism is preserved).
    df_in = pd.read_parquet(args.parquet)
    pulse_df = df_in[df_in["fidelity"] == "high"].sort_values("run_id").reset_index(drop=True)
    if len(pulse_df) != n_rows:
        raise RuntimeError(
            f"Pulse row count mismatch: parquet has {len(pulse_df)}, design has {n_rows}"
        )
    print(f"Reusing {len(pulse_df)} Pulse arm rows from existing parquet")

    pulse_records: list[RunRecord] = []
    feat_cols = [f"feat_{a}" for a in AXIS_NAMES]
    for i in range(n_rows):
        r = pulse_df.iloc[i]
        # Reconstruct features from feat_<axis> columns.
        feat_row = np.array([float(r[c]) for c in feat_cols], dtype=float)
        o2_raw = r["cerebral_o2_min"]
        o2 = None if pd.isna(o2_raw) else float(o2_raw)
        pulse_records.append(
            RunRecord(
                run_id=f"phase7_1b-pulse-{i:04d}",
                fidelity=Fidelity.HIGH,
                features=_row_to_features(feat_row),
                time_to_gloc_s=None,
                cerebral_o2_min=o2,
                engine_version="pulse-4.3.1",
            )
        )

    # Write the new paired parquet.
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    write_records_parquet([*cgem_records, *pulse_records], args.parquet)
    print(
        f"Wrote {len(cgem_records) + len(pulse_records)} records "
        f"({len(cgem_records)} CGEM rebuilt + {len(pulse_records)} Pulse preserved) "
        f"to {args.parquet}"
    )

    # Build the new manifest. Pulse arm metadata preserved verbatim; CGEM
    # arm metadata updated with new timing, output_channel, runtime info.
    cgem_fit_info = _capture_fit_info(CGEM_PARQUET)
    new_manifest = ProvenanceManifest(
        phase="7.1b",
        seed=args.seed,
        n_pulse_base=manifest_in.n_pulse_base,
        n_cgem_base=manifest_in.n_cgem_base,
        runtime={
            "python_version": platform.python_version(),
            "xgboost_version": __import__("xgboost").__version__,
            "hostname": socket.gethostname(),
            "rebuild_note": "CGEM arm rebuilt 2026-05-16 with predict_array channel; Pulse arm preserved",
        },
        design=DesignManifest(
            axis_names=list(axis_names),
            n_rows=n_rows,
            saltelli_calc_second_order=True,
        ),
        cgem=CgemArmManifest(
            training_parquet_path=manifest_in.cgem.training_parquet_path,
            training_parquet_sha256=training_sha,
            fit_info=cgem_fit_info,
            output_channel="predict_array",
            row_count=len(cgem_records),
            error_count=0,
            wall_clock_s=cgem_wall,
            started_at=cgem_started,
            finished_at=cgem_finished,
        ),
        pulse=PulseArmManifest(
            docker_image=manifest_in.pulse.docker_image,
            docker_image_digest=manifest_in.pulse.docker_image_digest,
            row_count=manifest_in.pulse.row_count,
            error_count=manifest_in.pulse.error_count,
            wall_clock_s=manifest_in.pulse.wall_clock_s,
            started_at=manifest_in.pulse.started_at,
            finished_at=manifest_in.pulse.finished_at,
        ),
    )
    write_manifest(new_manifest, args.manifest)
    print(f"Wrote rebuilt manifest to {args.manifest}")

    cgem_times = [r.time_to_gloc_s for r in cgem_records if r.time_to_gloc_s is not None]
    if cgem_times:
        print(
            f"\nCGEM regressor-only channel: "
            f"min={min(cgem_times):.4f} max={max(cgem_times):.4f} "
            f"mean={np.mean(cgem_times):.4f}"
        )


if __name__ == "__main__":
    main()
