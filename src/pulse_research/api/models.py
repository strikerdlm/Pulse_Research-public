"""Pydantic request and response schemas for the orchestration API."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ExperimentStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CreateExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128)
    n_base: int = Field(default=256, ge=2, le=4096)
    seed: int = Field(default=42)


class ExperimentSummary(BaseModel):
    id: UUID
    name: str
    n_base: int
    seed: int
    status: ExperimentStatus
    n_design_rows: int
    created_at: datetime


class ExperimentDetail(ExperimentSummary):
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error: str | None = None
    has_outputs: bool = False
    engine_label: str = ""
    failed_rows: int = Field(default=0, ge=0)


class StatusEvent(BaseModel):
    status: ExperimentStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    ts: datetime
    error: str | None = None


class SobolResponse(BaseModel):
    """Saltelli-Sobol indices returned by ``GET /experiments/{id}/sobol``."""

    names: list[str]
    S1: list[float]
    S1_conf: list[float]
    ST: list[float]
    ST_conf: list[float]
    S2: list[list[float]] | None = None
    S2_conf: list[list[float]] | None = None
    n_resamples: int
    seed: int
    st_stability: float


class ShapResponse(BaseModel):
    """SHAP attribution returned by ``GET /experiments/{id}/shap``."""

    feature_names: list[str]
    mean_abs: list[float]
    base_value: float
    train_mae: float
    values: list[list[float]] | None = None
    seed: int
