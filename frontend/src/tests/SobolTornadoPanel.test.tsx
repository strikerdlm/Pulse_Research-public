// frontend/src/tests/SobolTornadoPanel.test.tsx
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
      sobol: vi.fn(),
    },
  },
}));

import { api } from "../api/client";
import { SobolTornadoPanel } from "../components/SobolTornadoPanel";

function wrap(ui: ReactNode) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

const HAPPY_RESPONSE = {
  names: ["a", "b", "c"],
  S1: [0.1, 0.5, 0.3],
  S1_conf: [0.01, 0.02, 0.01],
  ST: [0.2, 0.7, 0.4],     // b > c > a
  ST_conf: [0.02, 0.03, 0.02],
  S2: null,
  S2_conf: null,
  n_resamples: 500,
  seed: 42,
  st_stability: 0.873,
};

beforeEach(() => {
  vi.mocked(api.experiments.sobol).mockReset();
});

describe("SobolTornadoPanel", () => {
  it("renders bars ranked by ST descending", async () => {
    vi.mocked(api.experiments.sobol).mockResolvedValue(HAPPY_RESPONSE);

    wrap(
      <SobolTornadoPanel
        experimentId="exp-1"
        status="completed"
        hasOutputs={true}
      />,
    );

    const chart = await waitFor(() => screen.getByTestId("echart"));
    const option = JSON.parse(chart.getAttribute("data-option") ?? "{}");
    expect(option.yAxis.data).toEqual(["b", "c", "a"]);
  });

  it("shows the st_stability score in the header", async () => {
    vi.mocked(api.experiments.sobol).mockResolvedValue(HAPPY_RESPONSE);

    wrap(
      <SobolTornadoPanel
        experimentId="exp-1"
        status="completed"
        hasOutputs={true}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/stability 0\.87/)).toBeInTheDocument();
    });
  });

  it("renders the empty-state message on 409 experiment_has_no_outputs", async () => {
    const { ApiError } = await import("../api/client");
    vi.mocked(api.experiments.sobol).mockRejectedValue(
      new ApiError(409, "experiment_has_no_outputs", "409 Conflict"),
    );

    wrap(
      <SobolTornadoPanel
        experimentId="exp-1"
        status="completed"
        hasOutputs={true}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByText(/Sobol indices unavailable/i),
      ).toBeInTheDocument();
    });
  });

  it("renders nothing when status is not completed", () => {
    wrap(
      <SobolTornadoPanel
        experimentId="exp-1"
        status="running"
        hasOutputs={false}
      />,
    );
    expect(screen.queryByText(/Saltelli-Sobol tornado/)).not.toBeInTheDocument();
  });
});
