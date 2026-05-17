import type {
  CreateExperimentRequest,
  ExperimentDataResponse,
  ExperimentDetail,
  ExperimentSummary,
  RunnerInfo,
  ShapQueryParams,
  ShapResponse,
  SobolQueryParams,
  SobolResponse,
} from "../types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let detail = text;
    try {
      const parsed = JSON.parse(text);
      if (parsed && typeof parsed.detail === "string") {
        detail = parsed.detail;
      }
    } catch {
      // body wasn't JSON; fall back to the raw text
    }
    throw new ApiError(res.status, detail, `${res.status} ${res.statusText}: ${text}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; version: string }>("/health"),

  runner: () => request<RunnerInfo>("/runner"),

  experiments: {
    list: () => request<ExperimentSummary[]>("/experiments"),
    get: (id: string) => request<ExperimentDetail>(`/experiments/${id}`),
    create: (body: CreateExperimentRequest) =>
      request<ExperimentSummary>("/experiments", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    run: (id: string) =>
      request<{ id: string; status: string }>(`/experiments/${id}/run`, {
        method: "POST",
      }),
    data: (id: string, opts: { sample?: number } = {}) => {
      const params = new URLSearchParams();
      if (opts.sample !== undefined) params.set("sample", String(opts.sample));
      const q = params.toString();
      const suffix = q ? `?${q}` : "";
      return request<ExperimentDataResponse>(`/experiments/${id}/data${suffix}`);
    },
    sobol: (id: string, params: SobolQueryParams = {}) => {
      const search = new URLSearchParams();
      if (params.num_resamples !== undefined) {
        search.set("num_resamples", String(params.num_resamples));
      }
      if (params.seed !== undefined) {
        search.set("seed", String(params.seed));
      }
      if (params.include_second_order !== undefined) {
        search.set(
          "include_second_order",
          params.include_second_order ? "true" : "false",
        );
      }
      const suffix = search.toString() ? `?${search.toString()}` : "";
      return request<SobolResponse>(`/experiments/${id}/sobol${suffix}`);
    },
    shap: (id: string, params: ShapQueryParams = {}) => {
      const search = new URLSearchParams();
      if (params.seed !== undefined) {
        search.set("seed", String(params.seed));
      }
      if (params.include_samples !== undefined) {
        search.set(
          "include_samples",
          params.include_samples ? "true" : "false",
        );
      }
      const q = search.toString();
      const suffix = q ? `?${q}` : "";
      return request<ShapResponse>(`/experiments/${id}/shap${suffix}`);
    },
    eventsUrl: (id: string) => `${BASE}/experiments/${id}/events`,
  },
};
