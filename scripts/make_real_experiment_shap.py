#!/usr/bin/env python3
"""End-to-end SHAP attribution from a SyntheticRunner experiment via the API.

Drives the full HTTP surface in-process (no live server, no Docker) and
writes an ECharts option JSON for the existing ``echarts`` skill to
render, plus the raw ``ShapResponse`` JSON.

Usage::

    python scripts/make_real_experiment_shap.py --n-base 1024 \\
        --name phase6_6_synthetic_n1024
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from pulse_research.api.app import create_app


def build_echarts_option(resp: dict[str, Any], title: str) -> dict[str, Any]:
    """ECharts horizontal-bar option ranked by mean(|SHAP|) descending."""
    mean_abs = np.asarray(resp["mean_abs"])
    order = np.argsort(mean_abs)[::-1]
    names = [resp["feature_names"][i] for i in order]
    data = [resp["mean_abs"][i] for i in order]

    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 140, "right": 40, "top": 70, "bottom": 40},
        "xAxis": {"type": "value", "name": "mean(|SHAP|)"},
        "yAxis": {"type": "category", "data": names, "inverse": False},
        "series": [
            {
                "name": "mean(|SHAP|)",
                "type": "bar",
                "data": data,
                "itemStyle": {"color": "#7fb4c9"},
            },
        ],
    }


async def _drive_api(n_base: int, seed: int) -> dict[str, Any]:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        created = (await ac.post(
            "/experiments",
            json={"name": "phase6_6_synthetic", "n_base": n_base, "seed": seed},
        )).json()
        exp_id = created["id"]
        assert (await ac.post(f"/experiments/{exp_id}/run")).status_code == 202
        status = "pending"
        for _ in range(20_000):
            await asyncio.sleep(0.05)
            status = (await ac.get(f"/experiments/{exp_id}")).json()["status"]
            if status in ("completed", "failed"):
                break
        assert status == "completed", f"unexpected status {status!r}"
        result: dict[str, Any] = (await ac.get(
            f"/experiments/{exp_id}/shap",
            params={"seed": seed},
        )).json()
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-base", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--name", type=str, default="phase6_6_synthetic_shap",
        help="Stem appended to YYYY-MM-DD_figure_<name>",
    )
    ap.add_argument(
        "--title", type=str,
        default="Phase 6.6 SHAP attribution (SyntheticRunner, N=1024)",
    )
    args = ap.parse_args()

    resp = asyncio.run(_drive_api(args.n_base, args.seed))
    option = build_echarts_option(resp, title=args.title)

    exports = Path("/root/repos/exports")
    exports.mkdir(parents=True, exist_ok=True)
    stem = f"{date.today():%Y-%m-%d}_figure_{args.name}"
    option_path = exports / f"{stem}_option.json"
    option_path.write_text(json.dumps(option, indent=2))

    raw_path = exports / f"{stem}_raw.json"
    raw_path.write_text(json.dumps(resp, indent=2))

    print(f"Surrogate MAE: {resp['train_mae']:.6f}")
    print(f"Wrote ECharts option to {option_path}")
    print(f"Wrote raw ShapResponse to {raw_path}")
    print(f"Render with: /echarts {option_path}")


if __name__ == "__main__":
    main()
