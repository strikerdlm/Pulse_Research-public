#!/usr/bin/env python3
"""Phase 7.1b audit script.

Verifies that a paired run-records parquet conforms to the spec's
"no fabricated data" invariants:

1. The training parquet referenced by the manifest matches its
   recorded SHA-256.
2. Every CGEM row in the output parquet is recomputable from a
   freshly-trained surrogate (``predict_array``, the regressor-only
   conditional-time channel) within ``rtol`` (default 1e-6). The
   audit re-uses ``make_cgem_surrogate_row_fn`` so the channel
   choice is centralized; switching channels in ``cgem_glue.py``
   automatically threads through.
3. Every Pulse row conforms to the schema contract:
   - finite ``cerebral_o2_min`` in ``(0, 1]`` and NaN ``time_to_gloc_s``
     when the row succeeded, OR
   - both fields NaN when the row errored.

Exits 0 if every check passes, 1 otherwise (with diff messages on
stdout).

Usage::

    python scripts/audit_run_records.py \\
        --parquet data/run_records_phase7_1b.parquet \\
        --manifest data/run_records_phase7_1b.manifest.json \\
        [--rtol 1e-6]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pulse_research.api.cgem_glue import make_cgem_surrogate_row_fn
from pulse_research.provenance.manifest import (
    ProvenanceManifest,
    compute_parquet_sha256,
)
from pulse_research.sensitivity.sobol_design import AXIS_NAMES


def _reconstruct_design_row(parquet_row: pd.Series) -> np.ndarray:
    """Rebuild the 11-axis design array from a parquet row's feat_<axis> columns."""
    return np.array(
        [float(parquet_row[f"feat_{name}"]) for name in AXIS_NAMES],
        dtype=float,
    )


def _check_pulse_schema_row(row: pd.Series) -> str | None:
    """Return None if the Pulse row conforms; an error message otherwise."""
    time_val = row.get("time_to_gloc_s")
    o2_val = row.get("cerebral_o2_min")
    time_nan = time_val is None or (
        isinstance(time_val, float) and math.isnan(time_val)
    )
    o2_nan = o2_val is None or (
        isinstance(o2_val, float) and math.isnan(o2_val)
    )

    # Pulse rows must always have NaN time_to_gloc_s (orthogonal oracle:
    # Pulse owns cerebral O2, not time-to-G-LOC).
    if not time_nan:
        return (
            f"pulse row {row.get('run_id')!r} has finite time_to_gloc_s="
            f"{time_val}; Pulse arm must not populate the time channel "
            f"(orthogonal oracle violation)"
        )

    if o2_nan:
        # Error row: both fields NaN is the contracted error pattern.
        return None

    # Success row: finite O2 must lie in (0, 1].
    try:
        o2_float = float(o2_val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"pulse row {row.get('run_id')!r} cerebral_o2_min not coercible to float: {o2_val!r}"
    if not (0.0 < o2_float <= 1.0):
        return (
            f"pulse row {row.get('run_id')!r} cerebral_o2_min={o2_float} "
            f"out of contracted range (0, 1]"
        )
    return None


def audit_run_records(
    parquet_path: Path,
    manifest_path: Path,
    *,
    rtol: float = 1e-6,
) -> tuple[int, list[str]]:
    """Audit a paired run-records parquet. Return (exit_code, messages).

    ``exit_code`` is 0 if every check passes, 1 otherwise. Messages
    describe any violations found.
    """
    msgs: list[str] = []
    parquet_path = Path(parquet_path)
    manifest_path = Path(manifest_path)

    # 1. Load manifest, validate schema.
    try:
        payload = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return 1, [f"manifest unreadable at {manifest_path}: {e}"]
    try:
        manifest = ProvenanceManifest.model_validate(payload)
    except Exception as e:
        return 1, [f"manifest failed schema validation: {e}"]

    # 2. Verify training parquet SHA-256.
    training_path = Path(manifest.cgem.training_parquet_path)
    try:
        observed_sha = compute_parquet_sha256(training_path)
    except (FileNotFoundError, PermissionError) as e:
        return 1, [
            f"training parquet at {training_path} unreadable for SHA-256 "
            f"verification: {e}"
        ]
    if observed_sha != manifest.cgem.training_parquet_sha256:
        msgs.append(
            f"training parquet SHA256 mismatch: manifest recorded "
            f"{manifest.cgem.training_parquet_sha256}, observed {observed_sha}"
        )

    # 3. Load the output parquet.
    try:
        df = pd.read_parquet(parquet_path)
    except (FileNotFoundError, OSError) as e:
        return 1, [*msgs, f"output parquet unreadable at {parquet_path}: {e}"]

    # If SHA-256 already mismatched, surrogate re-run would compare against
    # a different model than produced the parquet — abort here.
    if msgs:
        return 1, msgs

    # 4. Re-train the surrogate (deterministic given the training parquet).
    row_fn = make_cgem_surrogate_row_fn(training_path)

    # 5. CGEM rows: re-predict and compare.
    cgem_df = df[df["fidelity"] == "low"]
    max_diff = 0.0
    cgem_mismatches = 0
    for _, parquet_row in cgem_df.iterrows():
        design_row = _reconstruct_design_row(parquet_row)
        recomputed = row_fn(design_row).time_to_gloc_s
        stored = parquet_row["time_to_gloc_s"]
        if recomputed is None or stored is None or (
            isinstance(stored, float) and math.isnan(stored)
        ):
            msgs.append(
                f"cgem row {parquet_row['run_id']!r} has missing "
                f"time_to_gloc_s (stored={stored}, recomputed={recomputed})"
            )
            cgem_mismatches += 1
            continue
        diff = abs(float(stored) - float(recomputed))
        if diff > max_diff:
            max_diff = diff
        if diff > rtol:
            msgs.append(
                f"cgem row {parquet_row['run_id']!r} time_to_gloc_s diff "
                f"{diff:.3e} exceeds rtol={rtol:.0e} "
                f"(stored={stored}, recomputed={recomputed})"
            )
            cgem_mismatches += 1

    # 6. Pulse rows: schema check.
    pulse_df = df[df["fidelity"] == "high"]
    pulse_schema_failures = 0
    pulse_error_rows = 0
    for _, pulse_row in pulse_df.iterrows():
        violation = _check_pulse_schema_row(pulse_row)
        if violation is not None:
            msgs.append(violation)
            pulse_schema_failures += 1
        else:
            o2_val = pulse_row.get("cerebral_o2_min")
            if o2_val is None or (isinstance(o2_val, float) and math.isnan(o2_val)):
                pulse_error_rows += 1

    # 7. Summary line.
    summary = (
        f"audit summary: cgem_rows={len(cgem_df)} mismatches={cgem_mismatches} "
        f"max_diff={max_diff:.3e} | "
        f"pulse_rows={len(pulse_df)} schema_failures={pulse_schema_failures} "
        f"error_rows={pulse_error_rows}"
    )
    msgs.append(summary)

    if cgem_mismatches > 0 or pulse_schema_failures > 0:
        return 1, msgs
    return 0, msgs


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=1e-6)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    code, msgs = audit_run_records(args.parquet, args.manifest, rtol=args.rtol)
    for m in msgs:
        print(m)
    return code


if __name__ == "__main__":
    sys.exit(main())
