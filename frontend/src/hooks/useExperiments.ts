import {
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { api } from "../api/client";
import type { CreateExperimentRequest, ShapQueryParams, SobolQueryParams } from "../types";

const KEYS = {
  list: ["experiments"] as const,
  detail: (id: string) => ["experiments", id] as const,
  data: (id: string, sample: number) =>
    ["experiments", id, "data", sample] as const,
  health: ["health"] as const,
  sobol: (id: string, numResamples: number, seed: number) =>
    ["experiments", id, "sobol", numResamples, seed] as const,
  shap: (id: string, seed: number) =>
    ["experiments", id, "shap", seed] as const,
};

export function useHealth() {
  return useQuery({
    queryKey: KEYS.health,
    queryFn: api.health,
    refetchInterval: 10_000,
    staleTime: 5_000,
  });
}

export function useExperiments() {
  return useQuery({
    queryKey: KEYS.list,
    queryFn: api.experiments.list,
    refetchInterval: 4_000,
  });
}

export function useExperiment(id: string | null) {
  return useQuery({
    queryKey: id ? KEYS.detail(id) : ["experiments", "_none"],
    queryFn: () => api.experiments.get(id as string),
    enabled: Boolean(id),
    refetchInterval: 2_000,
  });
}

export function useCreateExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateExperimentRequest) => api.experiments.create(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: KEYS.list });
    },
  });
}

export function useRunExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.experiments.run(id),
    onSuccess: (_, id) => {
      qc.invalidateQueries({ queryKey: KEYS.list });
      qc.invalidateQueries({ queryKey: KEYS.detail(id) });
    },
  });
}

export function useExperimentData(
  id: string | null,
  { sample = 500 }: { sample?: number } = {},
) {
  return useQuery({
    queryKey: id ? KEYS.data(id, sample) : ["experiments", "_none", "data"],
    queryFn: () => api.experiments.data(id as string, { sample }),
    enabled: Boolean(id),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "completed" || status === "failed" ? false : 4_000;
    },
  });
}

export function useSobol(
  id: string,
  params: SobolQueryParams = {},
  options: { enabled?: boolean } = {},
) {
  const numResamples = params.num_resamples ?? 500;
  const seed = params.seed ?? 42;
  return useQuery({
    queryKey: KEYS.sobol(id, numResamples, seed),
    queryFn: () => api.experiments.sobol(id, params),
    enabled: options.enabled ?? true,
    staleTime: Infinity,  // indices are deterministic for fixed inputs
    retry: false,         // 409s are not transient
  });
}

export function useShap(
  id: string,
  params: ShapQueryParams = {},
  options: { enabled?: boolean } = {},
) {
  const seed = params.seed ?? 42;
  return useQuery({
    queryKey: KEYS.shap(id, seed),
    queryFn: () => api.experiments.shap(id, params),
    enabled: options.enabled ?? true,
    staleTime: Infinity,  // SHAP is deterministic for fixed (design, outputs, seed)
    retry: false,         // 409s are not transient
  });
}
