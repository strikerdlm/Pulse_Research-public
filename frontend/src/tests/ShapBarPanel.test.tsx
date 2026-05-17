// frontend/src/tests/ShapBarPanel.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("echarts-for-react", () => ({
  default: ({ option }: { option: Record<string, unknown> }) => (
    <div data-testid="echart" data-option={JSON.stringify(option)} />
  ),
}));

vi.mock("../api/client", () => ({
  ApiError: class extends Error {
    constructor(
      public status: number,
      public detail: string,
      message: string,
    ) {
      super(message);
      this.name = "ApiError";
    }
  },
  api: {
    experiments: {
      shap: vi.fn(),
    },
  },
}));

import { api } from "../api/client";
import { ShapBarPanel } from "../components/ShapBarPanel";

function wrap(ui: ReactNode) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

const HAPPY_RESPONSE = {
  feature_names: ["a", "b", "c"],
  mean_abs: [0.10, 0.50, 0.30],   // b > c > a
  base_value: 1.234,
  train_mae: 0.0123,
  values: null,
  seed: 42,
};

beforeEach(() => {
  vi.mocked(api.experiments.shap).mockReset();
});

describe("ShapBarPanel", () => {
  it("renders bars ranked by mean_abs descending", async () => {
    vi.mocked(api.experiments.shap).mockResolvedValue(HAPPY_RESPONSE);

    wrap(
      <ShapBarPanel
        experimentId="exp-1"
        status="completed"
        hasOutputs={true}
      />,
    );

    const chart = await waitFor(() => screen.getByTestId("echart"));
    const option = JSON.parse(chart.getAttribute("data-option") ?? "{}");
    expect(option.yAxis.data).toEqual(["b", "c", "a"]);
  });

  it("shows the surrogate MAE in the header", async () => {
    vi.mocked(api.experiments.shap).mockResolvedValue(HAPPY_RESPONSE);

    wrap(
      <ShapBarPanel
        experimentId="exp-1"
        status="completed"
        hasOutputs={true}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/surrogate MAE 0\.0123/)).toBeInTheDocument();
    });
  });

  it("renders the empty-state message on 409 experiment_has_no_outputs", async () => {
    const { ApiError } = await import("../api/client");
    vi.mocked(api.experiments.shap).mockRejectedValue(
      new ApiError(409, "experiment_has_no_outputs", "409 Conflict"),
    );

    wrap(
      <ShapBarPanel
        experimentId="exp-1"
        status="completed"
        hasOutputs={true}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByText(/SHAP attribution unavailable/i),
      ).toBeInTheDocument();
    });
  });

  it("renders nothing when status is not completed", () => {
    wrap(
      <ShapBarPanel
        experimentId="exp-1"
        status="running"
        hasOutputs={false}
      />,
    );
    expect(screen.queryByText(/SHAP attribution/)).not.toBeInTheDocument();
  });
});
