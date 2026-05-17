import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RunnerBadge } from "../components/RunnerBadge";

describe("RunnerBadge", () => {
  it("renders the SYNTHETIC label and glyph", () => {
    render(<RunnerBadge kind="synthetic" />);
    const badge = screen.getByTestId("runner-badge-synthetic");
    expect(badge).toHaveTextContent(/SYNTHETIC/);
    expect(badge).toHaveTextContent("▣");
  });

  it("renders the CGEM label and glyph", () => {
    render(<RunnerBadge kind="cgem" />);
    const badge = screen.getByTestId("runner-badge-cgem");
    expect(badge).toHaveTextContent(/CGEM/);
    expect(badge).toHaveTextContent("◈");
  });

  it("renders the PULSE label and glyph", () => {
    render(<RunnerBadge kind="pulse" />);
    const badge = screen.getByTestId("runner-badge-pulse");
    expect(badge).toHaveTextContent(/PULSE/);
    expect(badge).toHaveTextContent("◉");
  });

  it("falls back to synthetic styling for an unknown kind", () => {
    render(<RunnerBadge kind="totally-unknown" />);
    expect(screen.getByTestId("runner-badge-synthetic")).toBeInTheDocument();
  });

  it("respects a custom label", () => {
    render(<RunnerBadge kind="cgem" label="LOW-FID" />);
    expect(screen.getByText(/LOW-FID/)).toBeInTheDocument();
  });
});
