import type { ExperimentStatus } from "../types";

const STYLES: Record<ExperimentStatus, { color: string; glyph: string; label: string }> = {
  pending:   { color: "text-ink-faded border-rule",          glyph: "○", label: "PENDING"   },
  running:   { color: "text-signal border-signal-dim",       glyph: "◐", label: "RUNNING"   },
  completed: { color: "text-spec border-spec/40",            glyph: "●", label: "COMPLETED" },
  failed:    { color: "text-warn border-warn/40",            glyph: "✕", label: "FAILED"    },
};

interface Props {
  status: ExperimentStatus;
}

export function StatusPill({ status }: Props) {
  const s = STYLES[status];
  return (
    <span
      data-testid={`status-pill-${status}`}
      className={`inline-flex items-center gap-1.5 border px-2 py-[2px] font-mono text-[10px] tracking-[0.16em] ${s.color}`}
    >
      <span aria-hidden="true">{s.glyph}</span>
      {s.label}
    </span>
  );
}
