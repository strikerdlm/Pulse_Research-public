/**
 * Wire-format types mirroring `src/pulse_research/api/models.py`.
 *
 * Update both files together when the contract changes. A future Phase 5.5
 * can add openapi-typescript codegen if the API surface grows.
 */

export type ExperimentStatus = "pending" | "running" | "completed" | "failed";

export interface CreateExperimentRequest {
  name: string;
  n_base: number;
  seed: number;
}

export interface ExperimentSummary {
  id: string;
  name: string;
  n_base: number;
  seed: number;
  status: ExperimentStatus;
  n_design_rows: number;
  created_at: string;
}

export interface ExperimentDetail extends ExperimentSummary {
  progress: number;
  error: string | null;
  has_outputs: boolean;
  engine_label: string;
  failed_rows: number;
}

export interface StatusEvent {
  status: ExperimentStatus;
  progress: number;
  ts?: string;
  error?: string;
}

export interface ExperimentDataResponse {
  id: string;
  status: ExperimentStatus;
  n_design_rows: number;
  n_returned: number;
  axes: string[];
  rows: number[][];
  outputs: (number | null)[];
  output_range: { min: number | null; max: number | null };
}

export type RunnerKind = "synthetic" | "cgem" | "pulse";

export interface RunnerInfo {
  active_kind: RunnerKind;
  engine_label: string;
  available_kinds: RunnerKind[];
  cgem: { configured: boolean; root: string | null };
  pulse: { image: string; work_dir: string | null };
}

export type SobolResponse = {
  names: string[];
  S1: number[];
  S1_conf: number[];
  ST: number[];
  ST_conf: number[];
  S2: number[][] | null;
  S2_conf: number[][] | null;
  n_resamples: number;
  seed: number;
  st_stability: number;
};

export type SobolQueryParams = {
  num_resamples?: number;
  seed?: number;
  include_second_order?: boolean;
};

export type ShapResponse = {
  feature_names: string[];
  mean_abs: number[];
  base_value: number;
  train_mae: number;
  values: number[][] | null;
  seed: number;
};

export type ShapQueryParams = {
  seed?: number;
  include_samples?: boolean;
};
