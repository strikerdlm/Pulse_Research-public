"""CRUD + run endpoints for experiments."""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, status

from pulse_research.api.models import (
    CreateExperimentRequest,
    ExperimentDetail,
    ExperimentStatus,
    ExperimentSummary,
    ShapResponse,
    SobolResponse,
)
from pulse_research.api.store import Experiment
from pulse_research.explain.shap_attribute import shap_explain
from pulse_research.explain.surrogate import fit_xgb_surrogate
from pulse_research.sensitivity.analyze import analyze_design, st_stability
from pulse_research.sensitivity.sobol_design import AXIS_NAMES, build_design

router = APIRouter(prefix="/experiments", tags=["experiments"])

DEFAULT_DATA_SAMPLE = 500

# Background runner tasks are tracked here so they aren't garbage-collected
# mid-flight: asyncio.create_task only holds a weakref to the task, so without
# a strong reference the task can be cancelled by GC while still running.
_RUNNING_TASKS: set[asyncio.Task[None]] = set()


def _summary(exp: Experiment) -> ExperimentSummary:
    return ExperimentSummary(
        id=exp.id,
        name=exp.name,
        n_base=exp.n_base,
        seed=exp.seed,
        status=exp.status,
        n_design_rows=exp.design.shape[0],
        created_at=exp.created_at,
    )


def _detail(exp: Experiment) -> ExperimentDetail:
    return ExperimentDetail(
        id=exp.id,
        name=exp.name,
        n_base=exp.n_base,
        seed=exp.seed,
        status=exp.status,
        n_design_rows=exp.design.shape[0],
        created_at=exp.created_at,
        progress=exp.progress,
        error=exp.error,
        has_outputs=exp.outputs is not None,
        engine_label=exp.engine_label,
        failed_rows=exp.failed_rows,
    )


@router.post(
    "",
    response_model=ExperimentSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_experiment(
    req: CreateExperimentRequest,
    request: Request,
) -> ExperimentSummary:
    try:
        design, _ = build_design(n_base=req.n_base, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    store = request.app.state.store
    exp = store.create(req.name, req.n_base, req.seed, design)
    return _summary(exp)


@router.get("", response_model=list[ExperimentSummary])
async def list_experiments(request: Request) -> list[ExperimentSummary]:
    store = request.app.state.store
    return [_summary(e) for e in store.list()]


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(
    experiment_id: UUID,
    request: Request,
) -> ExperimentDetail:
    store = request.app.state.store
    exp = store.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return _detail(exp)


def _sample_design_and_outputs(
    exp: Experiment, sample: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (rows_sampled, outputs_sampled_or_None).

    Deterministic: same experiment.seed + same sample size → same row subset.
    If sample >= n, returns all rows in original order.
    """
    n = exp.design.shape[0]
    clamped = max(1, min(sample, n))
    if clamped >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(exp.seed)
        idx = np.sort(rng.choice(n, size=clamped, replace=False))
    rows = exp.design[idx]
    outputs = exp.outputs[idx] if exp.outputs is not None else None
    return rows, outputs


def _nan_to_null_list(values: np.ndarray) -> list[float | None]:
    """Project a 1-D float array to a JSON-safe list with NaN → None."""
    return [None if np.isnan(v) else float(v) for v in values]


@router.get("/{experiment_id}/data")
async def get_experiment_data(
    experiment_id: UUID,
    request: Request,
    sample: int = Query(default=DEFAULT_DATA_SAMPLE, ge=1, le=50_000),
) -> dict[str, Any]:
    """Sampled view of the design matrix + aligned outputs, plus axis names.

    Used by the frontend's parallel-coordinates plot. Returns a deterministic
    subset (seeded by the experiment's own seed) so repeat requests yield the
    same rows.
    """
    store = request.app.state.store
    exp = store.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experiment not found")

    rows, outputs = _sample_design_and_outputs(exp, sample)

    if outputs is None:
        outputs_list: list[float | None] = [None] * rows.shape[0]
        out_min: float | None = None
        out_max: float | None = None
    else:
        outputs_list = _nan_to_null_list(outputs)
        non_null = [v for v in outputs_list if v is not None]
        out_min = min(non_null) if non_null else None
        out_max = max(non_null) if non_null else None

    return {
        "id": str(exp.id),
        "status": exp.status.value,
        "n_design_rows": int(exp.design.shape[0]),
        "n_returned": int(rows.shape[0]),
        "axes": list(AXIS_NAMES),
        "rows": rows.tolist(),
        "outputs": outputs_list,
        "output_range": {"min": out_min, "max": out_max},
    }


@router.post(
    "/{experiment_id}/run",
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_experiment(
    experiment_id: UUID,
    request: Request,
) -> dict[str, Any]:
    store = request.app.state.store
    runner = request.app.state.runner
    exp = store.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    if exp.status is not ExperimentStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"experiment is {exp.status.value}; cannot start run",
        )
    task = asyncio.create_task(runner.run(experiment_id))
    _RUNNING_TASKS.add(task)
    task.add_done_callback(_RUNNING_TASKS.discard)
    return {"id": str(experiment_id), "status": "accepted"}


@router.get("/{experiment_id}/sobol", response_model=SobolResponse)
async def get_experiment_sobol(
    experiment_id: UUID,
    request: Request,
    num_resamples: int = Query(500, ge=1, le=5000),
    seed: int = Query(42, ge=0, le=2**32 - 1),
    include_second_order: bool = Query(False),
) -> SobolResponse:
    """Run the Saltelli-Sobol analyzer over the experiment's outputs."""
    store = request.app.state.store
    exp = store.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experiment_not_found")
    if exp.status != ExperimentStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="experiment_not_completed")
    if exp.outputs is None:
        raise HTTPException(status_code=409, detail="experiment_has_no_outputs")

    try:
        # The Saltelli design is always built with calc_second_order=True
        # (N * 24 rows). The analyzer must match; we suppress S2 in the
        # response when include_second_order=False.
        indices = analyze_design(
            exp.outputs,
            num_resamples=num_resamples,
            seed=seed,
            calc_second_order=True,
        )
    except ValueError as err:
        if "NaN" in str(err):
            raise HTTPException(
                status_code=409, detail="outputs_contain_nan"
            ) from err
        raise

    S2 = indices.S2.tolist() if include_second_order and indices.S2 is not None else None
    S2_conf = indices.S2_conf.tolist() if include_second_order and indices.S2_conf is not None else None

    return SobolResponse(
        names=indices.names,
        S1=indices.S1.tolist(),
        S1_conf=indices.S1_conf.tolist(),
        ST=indices.ST.tolist(),
        ST_conf=indices.ST_conf.tolist(),
        S2=S2,
        S2_conf=S2_conf,
        n_resamples=indices.n_resamples,
        seed=seed,
        st_stability=st_stability(indices),
    )


@router.get("/{experiment_id}/shap", response_model=ShapResponse)
async def get_experiment_shap(
    experiment_id: UUID,
    request: Request,
    seed: int = Query(42, ge=0, le=2**32 - 1),
    include_samples: bool = Query(False),
) -> ShapResponse:
    """Fit an XGBoost surrogate to the experiment outputs and run SHAP."""
    store = request.app.state.store
    exp = store.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experiment_not_found")
    if exp.status != ExperimentStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="experiment_not_completed")
    if exp.outputs is None:
        raise HTTPException(status_code=409, detail="experiment_has_no_outputs")
    if np.isnan(exp.outputs).any():
        raise HTTPException(status_code=409, detail="outputs_contain_nan")

    surrogate = fit_xgb_surrogate(
        exp.design, exp.outputs, list(AXIS_NAMES), seed=seed,
    )
    attribution = shap_explain(surrogate, exp.design)

    values = attribution.values.tolist() if include_samples else None

    return ShapResponse(
        feature_names=attribution.feature_names,
        mean_abs=attribution.mean_abs.tolist(),
        base_value=attribution.base_value,
        train_mae=surrogate.train_mae,
        values=values,
        seed=seed,
    )
