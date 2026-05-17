import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";

const RUNNER_KEY = ["runner"] as const;

export function useRunnerInfo() {
  return useQuery({
    queryKey: RUNNER_KEY,
    queryFn: api.runner,
    refetchOnWindowFocus: true,
    staleTime: 30_000,
  });
}
