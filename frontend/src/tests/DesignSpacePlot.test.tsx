import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DesignSpacePlot } from "../components/DesignSpacePlot";

// echarts-for-react renders a canvas inside an SVG; in happy-dom the only
// reliable assertion is that the component mounted and the title bar exists.
// We do not assert on the chart's internal DOM — that would couple this test
// to the rendering backend's implementation details.

vi.mock("echarts-for-react", () => ({
  default: ({ option }: { option: unknown }) => (
    <div data-testid="echarts-mock" data-has-visualmap={
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (option as any).visualMap !== undefined ? "true" : "false"
    } />
  ),
}));

const AXES = [
  "gz_peak", "gz_onset_rate", "seat_tilt_deg", "anti_g_strain",
  "pilot_weight_kg", "pilot_height_cm", "pilot_age_y",
  "baseline_vo2max", "baseline_map_mmhg",
  "fio2_inspired", "sao2_baseline",
];

const SAMPLE_ROWS = [
  [6, 3, 15, 0.5, 80, 175, 35, 50, 90, 0.21, 0.97],
  [4, 2, 10, 0.3, 70, 170, 30, 45, 85, 0.18, 0.93],
  [8, 4, 20, 0.8, 90, 180, 40, 55, 100, 0.30, 0.99],
];

describe("DesignSpacePlot", () => {
  it("renders the title and row count subtitle", () => {
    render(
      <DesignSpacePlot
        rows={SAMPLE_ROWS}
        outputs={[null, null, null]}
        axes={AXES}
        outputRange={{ min: null, max: null }}
      />,
    );
    expect(screen.getByText(/design space/i)).toBeInTheDocument();
    expect(screen.getByText(/3 rows/)).toBeInTheDocument();
  });

  it("omits the visualMap when no output range is provided", () => {
    render(
      <DesignSpacePlot
        rows={SAMPLE_ROWS}
        outputs={[null, null, null]}
        axes={AXES}
        outputRange={{ min: null, max: null }}
      />,
    );
    expect(screen.getByTestId("echarts-mock")).toHaveAttribute(
      "data-has-visualmap",
      "false",
    );
  });

  it("adds a visualMap when an output range is provided", () => {
    render(
      <DesignSpacePlot
        rows={SAMPLE_ROWS}
        outputs={[12.5, 14.0, 9.2]}
        axes={AXES}
        outputRange={{ min: 9.2, max: 14.0 }}
      />,
    );
    expect(screen.getByTestId("echarts-mock")).toHaveAttribute(
      "data-has-visualmap",
      "true",
    );
  });

  it("uses the provided subtitle when supplied", () => {
    render(
      <DesignSpacePlot
        rows={SAMPLE_ROWS}
        outputs={[null, null, null]}
        axes={AXES}
        outputRange={{ min: null, max: null }}
        subtitle="500 / 1536 rows · seed 42"
      />,
    );
    expect(screen.getByText(/500 \/ 1536 rows/)).toBeInTheDocument();
  });
});
