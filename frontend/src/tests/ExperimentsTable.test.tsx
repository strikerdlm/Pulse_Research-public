import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ExperimentsTable } from "../components/ExperimentsTable";
import type { ExperimentSummary } from "../types";

const FIXTURE: ExperimentSummary[] = [
  {
    id: "11111111-1111-1111-1111-111111111111",
    name: "baseline-21pct-o2",
    n_base: 64,
    seed: 42,
    status: "pending",
    n_design_rows: 1536,
    created_at: "2026-05-14T10:00:00Z",
  },
  {
    id: "22222222-2222-2222-2222-222222222222",
    name: "altitude-25k",
    n_base: 128,
    seed: 7,
    status: "completed",
    n_design_rows: 3072,
    created_at: "2026-05-14T11:30:00Z",
  },
];

describe("ExperimentsTable", () => {
  it("renders the empty state when there are no experiments", () => {
    render(
      <ExperimentsTable experiments={[]} selectedId={null} onSelect={() => {}} />,
    );
    expect(screen.getByTestId("experiments-empty")).toBeInTheDocument();
    expect(screen.getByText(/no flights logged/i)).toBeInTheDocument();
  });

  it("renders one row per experiment", () => {
    render(
      <ExperimentsTable
        experiments={FIXTURE}
        selectedId={null}
        onSelect={() => {}}
      />,
    );
    expect(screen.getByText("baseline-21pct-o2")).toBeInTheDocument();
    expect(screen.getByText("altitude-25k")).toBeInTheDocument();
    expect(screen.getByText("1536")).toBeInTheDocument();
    expect(screen.getByText("3072")).toBeInTheDocument();
  });

  it("calls onSelect with the row id when clicked", () => {
    const onSelect = vi.fn();
    render(
      <ExperimentsTable
        experiments={FIXTURE}
        selectedId={null}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByText("baseline-21pct-o2"));
    expect(onSelect).toHaveBeenCalledWith(FIXTURE[0].id);
  });

  it("renders a status pill for each row", () => {
    render(
      <ExperimentsTable
        experiments={FIXTURE}
        selectedId={null}
        onSelect={() => {}}
      />,
    );
    expect(screen.getByTestId("status-pill-pending")).toBeInTheDocument();
    expect(screen.getByTestId("status-pill-completed")).toBeInTheDocument();
  });
});
