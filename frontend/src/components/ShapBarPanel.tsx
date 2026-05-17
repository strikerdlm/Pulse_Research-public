// frontend/src/components/ShapBarPanel.tsx
import ReactECharts from "echarts-for-react";
import { useMemo } from "react";

import { ApiError } from "../api/client";
import { useShap } from "../hooks/useExperiments";
import type { ExperimentStatus, ShapResponse } from "../types";

type ShapBarPanelProps = {
  experimentId: string;
  status: ExperimentStatus;
  hasOutputs: boolean;
};

const COLOR_BAR = "#7fb4c9";

function buildOption(resp: ShapResponse) {
  const order = resp.mean_abs
    .map((v, i) => [v, i] as const)
    .sort((a, b) => b[0] - a[0])
    .map(([, i]) => i);
  const names = order.map((i) => resp.feature_names[i]);
  const data = order.map((i) => resp.mean_abs[i]);

  return {
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    grid: { left: 140, right: 40, top: 40, bottom: 40 },
    xAxis: { type: "value", name: "mean(|SHAP|)" },
    yAxis: { type: "category", data: names, inverse: false },
    series: [
      {
        name: "mean(|SHAP|)",
        type: "bar",
        data,
        itemStyle: { color: COLOR_BAR },
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
      return "Experiment completed without outputs — SHAP attribution unavailable.";
    }
    if (error.detail === "outputs_contain_nan") {
      return "Outputs contain NaN — XGBoost surrogate cannot fit. Re-execute the experiment.";
    }
  }
  return "Failed to load SHAP attribution.";
}

export function ShapBarPanel({
  experimentId,
  status,
  hasOutputs,
}: ShapBarPanelProps) {
  const enabled = status === "completed" && hasOutputs;
  const { data, isPending, error } = useShap(
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
      aria-labelledby="shap-panel-heading"
      className="mt-6"
    >
      <header className="flex items-baseline justify-between mb-2">
        <h2 id="shap-panel-heading" className="m-0 font-mono text-lg text-ink">
          SHAP attribution
        </h2>
        {data ? (
          <span
            title="XGBoost surrogate distillation MAE on the training cohort. Lower = SHAP more faithful to the runner output."
            className="font-mono text-xs text-ink-faded px-2 py-0.5 border border-rule rounded"
          >
            surrogate MAE {data.train_mae.toFixed(4)}
          </span>
        ) : null}
      </header>

      {isPending ? (
        <p className="text-ink-faded font-mono text-sm">
          Fitting XGBoost surrogate and computing SHAP…
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="text-warn font-mono text-sm">
          {emptyMessage(error)}
        </p>
      ) : null}
      {option ? (
        <ReactECharts option={option} style={{ height: 380 }} />
      ) : null}
    </section>
  );
}
