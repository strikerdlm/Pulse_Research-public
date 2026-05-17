import type { ExperimentSummary } from "../types";
import { StatusPill } from "./StatusPill";

interface Props {
  experiments: ExperimentSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

function fmtTime(iso: string): string {
  const d = new Date(iso);
  return d.toISOString().replace("T", " ").slice(0, 19) + "Z";
}

export function ExperimentsTable({ experiments, selectedId, onSelect }: Props) {
  if (experiments.length === 0) {
    return (
      <div className="border border-rule p-12 text-center" data-testid="experiments-empty">
        <p className="font-display italic text-2xl text-ink-faded mb-2">
          no flights logged.
        </p>
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-quiet">
          create the first experiment to populate the register.
        </p>
      </div>
    );
  }

  return (
    <div
      data-testid="experiments-table"
      className="border border-rule bg-panel"
    >
      <table className="w-full font-mono text-[12px]">
        <thead className="text-[10px] uppercase tracking-[0.18em] text-ink-faded border-b border-rule">
          <tr>
            <Th>id</Th>
            <Th>name</Th>
            <Th align="right">n_base</Th>
            <Th align="right">rows</Th>
            <Th align="right">seed</Th>
            <Th>status</Th>
            <Th>created</Th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((e) => {
            const isSelected = e.id === selectedId;
            return (
              <tr
                key={e.id}
                onClick={() => onSelect(e.id)}
                className={`border-b border-rule/50 cursor-pointer transition-colors ${
                  isSelected ? "bg-signal/[0.06]" : "hover:bg-panel-2"
                }`}
              >
                <Td>
                  <span
                    className={`tabular-nums ${
                      isSelected ? "text-signal" : "text-ink-faded"
                    }`}
                  >
                    {e.id.slice(0, 8)}
                  </span>
                </Td>
                <Td>
                  <span className={isSelected ? "text-ink" : "text-ink/90"}>
                    {e.name}
                  </span>
                </Td>
                <Td align="right" mono>{e.n_base}</Td>
                <Td align="right" mono>{e.n_design_rows}</Td>
                <Td align="right" mono>{e.seed}</Td>
                <Td><StatusPill status={e.status} /></Td>
                <Td>
                  <span className="text-ink-faded text-[10px]">
                    {fmtTime(e.created_at)}
                  </span>
                </Td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Th({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th
      className={`px-3 py-2 font-normal ${
        align === "right" ? "text-right" : "text-left"
      }`}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  mono,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  mono?: boolean;
}) {
  return (
    <td
      className={`px-3 py-2 ${align === "right" ? "text-right" : "text-left"} ${
        mono ? "tabular-nums text-ink" : ""
      }`}
    >
      {children}
    </td>
  );
}
