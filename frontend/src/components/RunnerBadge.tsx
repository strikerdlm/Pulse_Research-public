import type { RunnerKind } from "../types";

const STYLES: Record<RunnerKind, { color: string; glyph: string; label: string }> = {
  synthetic: { color: "text-ink-faded border-rule",      glyph: "▣", label: "SYNTHETIC" },
  cgem:      { color: "text-trace border-trace/40",      glyph: "◈", label: "CGEM"      },
  pulse:     { color: "text-signal border-signal/60",    glyph: "◉", label: "PULSE"     },
};

interface Props {
  kind: RunnerKind | string;
  label?: string;
}

/**
 * Engine badge: signal-coloured chip identifying which runner produced (or
 * is producing) an experiment's outputs. Used both in MissionControl and on
 * the ExperimentDetail header.
 */
export function RunnerBadge({ kind, label }: Props) {
  const normalized = (kind in STYLES ? kind : "synthetic") as RunnerKind;
  const s = STYLES[normalized];
  return (
    <span
      data-testid={`runner-badge-${normalized}`}
      className={`inline-flex items-center gap-1.5 border px-2 py-[2px] font-mono text-[10px] tracking-[0.18em] ${s.color}`}
    >
      <span aria-hidden="true">{s.glyph}</span>
      {label ?? s.label}
    </span>
  );
}
