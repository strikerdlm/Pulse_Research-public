interface Props {
  label?: string;
  align?: "left" | "right";
}

export function HairlineRule({ label, align = "left" }: Props) {
  return (
    <div className="flex items-center gap-3 my-3 text-ink-faded">
      {align === "right" && <div className="flex-1 h-px bg-rule" />}
      {label && (
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faded">
          {label}
        </span>
      )}
      {align === "left" && <div className="flex-1 h-px bg-rule" />}
    </div>
  );
}
