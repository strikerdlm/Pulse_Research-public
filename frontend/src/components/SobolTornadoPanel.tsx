// frontend/src/components/SobolTornadoPanel.tsx
import ReactECharts from "echarts-for-react";
import { useMemo } from "react";

import { ApiError } from "../api/client";
import { useSobol } from "../hooks/useExperiments";
import type { ExperimentStatus, SobolResponse } from "../types";

type SobolTornadoPanelProps = {
  experimentId: string;
  status: ExperimentStatus;
  hasOutputs: boolean;
};

const COLOR_S1 = "#7fb4c9";
const COLOR_ST = "#d4a657";

function buildOption(resp: SobolResponse) {
  const order = resp.ST
    .map((v, i) => [v, i] as const)
    .sort((a, b) => b[0] - a[0])
    .map(([, i]) => i);
  const names = order.map((i) => resp.names[i]);
  const S1 = order.map((i) => resp.S1[i]);
  const ST = order.map((i) => resp.ST[i]);

  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    legend: { data: ["S1 (first-order)", "ST (total)"], top: 0 },
    grid: { left: 140, right: 40, top: 40, bottom: 40 },
    xAxis: { type: "value", name: "Sobol index" },
    yAxis: { type: "category", data: names, inverse: false },
    series: [
      {
        name: "S1 (first-order)",
        type: "bar",
        data: S1,
        itemStyle: { color: COLOR_S1 },
      },
      {
        name: "ST (total)",
        type: "bar",
        data: ST,
        itemStyle: { color: COLOR_ST },
      },
    ],
  };
}

function emptyMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.detail === "experiment_not_completed") {
      return "Experiment is still running.";
    }
    if (error.detail === "experiment_has_no_outputs") {
      return "Experiment completed without outputs — Sobol indices unavailable.";
    }
    if (error.detail === "outputs_contain_nan") {
      return "Outputs contain NaN — analyzer cannot run. Re-execute the experiment.";
    }
  }
  return "Failed to load Sobol indices.";
}

export function SobolTornadoPanel({
  experimentId,
  status,
  hasOutputs,
}: SobolTornadoPanelProps) {
  const enabled = status === "completed" && hasOutputs;
  const { data, isPending, error } = useSobol(
    experimentId,
    {},
    { enabled },
  );

  const option = useMemo(() => (data ? buildOption(data) : null), [data]);

  if (!enabled) {
    return null;
  }

  return (
    <section
      aria-labelledby="sobol-panel-heading"
      className="mt-6"
    >
      <header className="flex items-baseline justify-between mb-2">
        <h2 id="sobol-panel-heading" className="m-0 font-mono text-lg text-ink">
          Saltelli-Sobol tornado
        </h2>
        {data ? (
          <span
            title="Bootstrap-CI relative width (Sarrazin 2016). 1.0 = perfectly tight CIs; ≥ 0.80 acceptable at production N."
            className="font-mono text-xs text-ink-faded px-2 py-0.5 border border-rule rounded"
          >
            stability {data.st_stability.toFixed(2)}
          </span>
        ) : null}
      </header>

      {isPending ? <p className="text-ink-faded font-mono text-sm">Computing Sobol indices…</p> : null}
      {error ? <p role="alert" className="text-warn font-mono text-sm">{emptyMessage(error)}</p> : null}
      {option ? (
        <ReactECharts option={option} style={{ height: 380 }} />
      ) : null}
    </section>
  );
}
