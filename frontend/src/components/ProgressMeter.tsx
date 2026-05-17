interface Props {
  /** 0..1 */
  value: number;
  label?: string;
  variant?: "running" | "done" | "fail" | "idle";
}

/**
 * Instrument-cluster horizontal progress bar. 20 tick marks; the active range
 * is filled in the variant color; numeric percent readout uses tabular figures.
 */
export function ProgressMeter({ value, label = "PROGRESS", variant = "running" }: Props) {
  const clamped = Math.max(0, Math.min(1, value));
  const pct = Math.round(clamped * 100);
  const segments = 20;
  const filled = Math.round(clamped * segments);

  const accent =
    variant === "done"
      ? "bg-spec"
      : variant === "fail"
        ? "bg-warn"
        : variant === "idle"
          ? "bg-rule-bright"
          : "bg-signal";

  return (
    <div className="flex items-center gap-4 select-none" data-testid="progress-meter">
      <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faded">
        {label}
      </span>
      <div className="flex-1 flex gap-[2px] items-center">
        {Array.from({ length: segments }).map((_, i) => (
          <span
            key={i}
            className={`flex-1 h-[10px] border border-rule ${
              i < filled ? accent : "bg-transparent"
            }`}
          />
        ))}
      </div>
      <span className="font-mono text-ink tabular-nums text-[12px]">
        {String(pct).padStart(3, " ")}<span className="text-ink-faded">%</span>
      </span>
    </div>
  );
}
