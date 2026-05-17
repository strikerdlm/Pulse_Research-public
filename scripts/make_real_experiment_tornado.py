#!/usr/bin/env python3
"""End-to-end Sobol tornado from a SyntheticRunner experiment via the API.

Drives the full HTTP surface in-process (no live server, no Docker) and
writes an ECharts option JSON for the existing ``echarts`` skill to
render.

Usage::

    python scripts/make_real_experiment_tornado.py --n-base 1024 \\
        --name phase6_5_synthetic_n1024
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np

from pulse_research.api.app import create_app


def build_echarts_option(resp: dict[str, object], title: str) -> dict[str, object]:
    """ECharts horizontal grouped-bar option ranked by ST descending."""
    ST = np.asarray(resp["ST"])
    order = np.argsort(ST)[::-1]
    names = [resp["names"][i] for i in order]  # type: ignore[index]
    S1 = [resp["S1"][i] for i in order]  # type: ignore[index]
    ST_sorted = [resp["ST"][i] for i in order]  # type: ignore[index]

    return {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": ["S1 (first-order)", "ST (total)"], "top": 30},
        "grid": {"left": 140, "right": 40, "top": 70, "bottom": 40},
        "xAxis": {"type": "value", "name": "Sobol index"},
        "yAxis": {"type": "category", "data": names, "inverse": False},
        "series": [
            {
                "name": "S1 (first-order)", "type": "bar",
                "data": S1, "itemStyle": {"color": "#7fb4c9"},
            },
            {
                "name": "ST (total)", "type": "bar",
                "data": ST_sorted, "itemStyle": {"color": "#d4a657"},
            },
        ],
    }


async def _drive_api(n_base: int, num_resamples: int, seed: int) -> dict[str, object]:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        created = (await ac.post(
            "/experiments",
            json={"name": "phase6_5_synthetic", "n_base": n_base, "seed": seed},
        )).json()
        exp_id = created["id"]
        assert (await ac.post(f"/experiments/{exp_id}/run")).status_code == 202
        status = "unknown"
        for _ in range(20_000):
            await asyncio.sleep(0.05)
            status = (await ac.get(f"/experiments/{exp_id}")).json()["status"]
            if status in ("completed", "failed"):
                break
        assert status == "completed", f"unexpected status {status!r}"
        result: dict[str, object] = (await ac.get(
            f"/experiments/{exp_id}/sobol",
            params={"num_resamples": num_resamples, "seed": seed},
        )).json()
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-base", type=int, default=1024)
    ap.add_argument("--num-resamples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--name", type=str, default="phase6_5_synthetic_tornado",
        help="Stem appended to YYYY-MM-DD_figure_<name>",
    )
    ap.add_argument(
        "--title", type=str,
        default="Phase 6.5 Sobol tornado (SyntheticRunner, N=1024)",
    )
    args = ap.parse_args()

    resp = asyncio.run(_drive_api(args.n_base, args.num_resamples, args.seed))
    option = build_echarts_option(resp, title=args.title)

    exports = Path("/root/repos/exports")
    exports.mkdir(parents=True, exist_ok=True)
    stem = f"{date.today():%Y-%m-%d}_figure_{args.name}"
    option_path = exports / f"{stem}_option.json"
    option_path.write_text(json.dumps(option, indent=2))

    raw_path = exports / f"{stem}_raw.json"
    raw_path.write_text(json.dumps(resp, indent=2))

    print(f"ST stability: {resp['st_stability']:.4f}")
    print(f"Wrote ECharts option to {option_path}")
    print(f"Wrote raw SobolResponse to {raw_path}")
    print(f"Render with: /echarts {option_path}")


if __name__ == "__main__":
    main()
