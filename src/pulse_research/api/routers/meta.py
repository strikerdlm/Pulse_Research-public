"""Operational introspection endpoints."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request

from pulse_research.api.cgem_glue import CGEM_ENV_VAR
from pulse_research.api.pulse_glue import (
    DEFAULT_PULSE_IMAGE,
    PULSE_IMAGE_ENV_VAR,
    PULSE_WORK_DIR_ENV_VAR,
)
from pulse_research.api.runners import AVAILABLE_RUNNER_KINDS

router = APIRouter(tags=["meta"])


@router.get("/runner")
async def get_runner_info(request: Request) -> dict[str, Any]:
    """Report which runner is wired into ``app.state`` plus oracle config state.

    Operators use this to confirm that ``PULSE_RESEARCH_RUNNER=<kind>`` took
    effect at startup, that ``CGEM_ROOT`` resolves to the upstream repo (for
    ``kind=cgem``), and that ``PULSE_DOCKER_IMAGE`` / ``PULSE_WORK_DIR`` are
    configured as expected (for ``kind=pulse``).
    """
    runner = request.app.state.runner
    cgem_root = os.environ.get(CGEM_ENV_VAR)
    pulse_image = os.environ.get(PULSE_IMAGE_ENV_VAR, DEFAULT_PULSE_IMAGE)
    pulse_work_dir = os.environ.get(PULSE_WORK_DIR_ENV_VAR)
    return {
        "active_kind": runner.engine_label,
        "engine_label": runner.engine_label,
        "cgem": {
            "configured": cgem_root is not None,
            "root": cgem_root,
        },
        "pulse": {
            "image": pulse_image,
            "work_dir": pulse_work_dir,
        },
        "available_kinds": list(AVAILABLE_RUNNER_KINDS),
    }
