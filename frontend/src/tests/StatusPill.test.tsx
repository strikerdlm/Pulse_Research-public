import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusPill } from "../components/StatusPill";

describe("StatusPill", () => {
  it("renders the PENDING label and glyph", () => {
    render(<StatusPill status="pending" />);
    const pill = screen.getByTestId("status-pill-pending");
    expect(pill).toHaveTextContent(/PENDING/);
    expect(pill).toHaveTextContent("○");
  });

  it("renders the COMPLETED label and glyph", () => {
    render(<StatusPill status="completed" />);
    const pill = screen.getByTestId("status-pill-completed");
    expect(pill).toHaveTextContent(/COMPLETED/);
    expect(pill).toHaveTextContent("●");
  });

  it("renders the FAILED label and glyph", () => {
    render(<StatusPill status="failed" />);
    const pill = screen.getByTestId("status-pill-failed");
    expect(pill).toHaveTextContent(/FAILED/);
    expect(pill).toHaveTextContent("✕");
  });

  it("renders the RUNNING label and glyph", () => {
    render(<StatusPill status="running" />);
    const pill = screen.getByTestId("status-pill-running");
    expect(pill).toHaveTextContent(/RUNNING/);
    expect(pill).toHaveTextContent("◐");
  });
});
