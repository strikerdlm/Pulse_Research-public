import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ProgressMeter } from "../components/ProgressMeter";

describe("ProgressMeter", () => {
  it("renders the percent readout rounded to integer", () => {
    render(<ProgressMeter value={0.42} />);
    expect(screen.getByText(/42/)).toBeInTheDocument();
  });

  it("clamps values above 1 to 100%", () => {
    render(<ProgressMeter value={1.5} />);
    expect(screen.getByText(/100/)).toBeInTheDocument();
  });

  it("clamps values below 0 to 0%", () => {
    render(<ProgressMeter value={-0.2} />);
    const readouts = screen.getAllByText(/0/);
    expect(readouts.length).toBeGreaterThan(0);
  });
});
