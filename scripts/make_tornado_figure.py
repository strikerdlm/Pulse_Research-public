#!/usr/bin/env python3
"""Generate an ECharts tornado-plot option JSON for a Saltelli analysis.

The script computes S1 and ST with bootstrap CIs on a 1-D output vector
(``.npy`` file) aligned with ``build_design``, then writes an ECharts
option JSON to ``/root/repos/exports/`` per the workspace file-output
rule.

Render the JSON to SVG/PNG by invoking the ``echarts`` skill on the
produced file path; the skill handles the headless render.

Usage::

    python scripts/make_tornado_figure.py outputs.npy \\
        --num-resamples 500 --seed 42 --name phase6_pulse_v1
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from pulse_research.sensitivity.analyze import SobolIndices, analyze_design


def build_echarts_option(idx: SobolIndices, title: str) -> dict[str, Any]:
    """ECharts horizontal grouped-bar option, ranked by ST descending."""
    order = np.argsort(idx.ST)[::-1]
    names = [idx.names[i] for i in order]
    S1 = idx.S1[order].tolist()
    ST = idx.ST[order].tolist()

    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["S1 (first-order)", "ST (total)"], "top": 30},
        "grid": {"left": 140, "right": 40, "top": 70, "bottom": 40},
        "xAxis": {"type": "value", "name": "Sobol index"},
        "yAxis": {"type": "category", "data": names, "inverse": False},
        "series": [
            {
                "name": "S1 (first-order)",
                "type": "bar",
                "data": S1,
                "itemStyle": {"color": "#7fb4c9"},
            },
            {
                "name": "ST (total)",
                "type": "bar",
                "data": ST,
                "itemStyle": {"color": "#d4a657"},
            },
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "outputs", type=Path,
        help="Path to a .npy file with the 1-D output vector",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-resamples", type=int, default=500)
    ap.add_argument(
        "--name", type=str, default="phase6_tornado",
        help="Stem appended to YYYY-MM-DD_figure_<name>",
    )
    ap.add_argument(
        "--title", type=str,
        default="Saltelli-Sobol tornado (CGEM-Pulse hypoxia surrogate)",
    )
    args = ap.parse_args()

    y = np.load(args.outputs)
    idx = analyze_design(
        y, num_resamples=args.num_resamples, seed=args.seed,
    )
    option = build_echarts_option(idx, title=args.title)

    exports = Path("/root/repos/exports")
    exports.mkdir(parents=True, exist_ok=True)
    stem = f"{date.today():%Y-%m-%d}_figure_{args.name}"
    option_path = exports / f"{stem}_option.json"
    option_path.write_text(json.dumps(option, indent=2))

    print(f"Wrote ECharts option to {option_path}")
    print("Render with: /echarts " + str(option_path))


if __name__ == "__main__":
    main()
