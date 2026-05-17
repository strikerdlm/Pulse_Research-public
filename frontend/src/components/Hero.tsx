/**
 * Editorial-scale title panel. Newsreader display with optical-size variation
 * and a SOFT axis pulled positive; subdescriptor in monospace small-caps;
 * decorative coordinate band below.
 */
export function Hero() {
  return (
    <section className="relative grain border-b border-rule">
      <div className="px-6 py-12 max-w-[1400px]">
        <p className="reveal font-mono text-[10px] uppercase tracking-[0.32em] text-ink-faded mb-6">
          paper // 02 · cgem × pulse · multi-fidelity surrogate
        </p>

        <h1
          className="reveal font-display font-light text-ink tracking-[-0.02em]"
          style={{
            fontSize: "clamp(2.5rem, 6vw, 5.5rem)",
            lineHeight: 0.95,
            fontVariationSettings: '"opsz" 72, "SOFT" 50, "WONK" 1',
          }}
        >
          G-LOC tolerance under
          <br />
          <span className="text-signal italic">acceleration</span> ×{" "}
          <span className="text-signal italic">hypoxia.</span>
        </h1>

        <p
          className="reveal mt-8 max-w-[48ch] text-ink-faded text-[13px] leading-[1.65] font-display"
          style={{ animationDelay: "120ms" }}
        >
          An orthogonal-oracle Gaussian-process surrogate composed of the FAA
          CAMI G-Effects Model (acceleration arm) and the Pulse Physiology
          Engine (hypoxia arm), coupled multiplicatively through the Hüfner
          arterial-oxygen-content equation without fitted parameters. Designed
          and operated from this console.
        </p>

        <dl
          className="reveal mt-10 grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-3 text-[10px] uppercase tracking-[0.18em]"
          style={{ animationDelay: "240ms" }}
        >
          <Coord label="axes" value="11" suffix="saltelli" />
          <Coord label="arms" value="02" suffix="cgem · pulse" />
          <Coord label="α" value="0.10" suffix="nominal" />
          <Coord label="cov." value="0.907" suffix="empirical" />
        </dl>
      </div>

      {/* Decorative coordinate band */}
      <div className="px-6 pb-3 font-mono text-[9px] text-ink-quiet uppercase tracking-[0.28em] flex justify-between border-t border-rule pt-3">
        <span>04° 42′ 50″ N / 74° 08′ 35″ W</span>
        <span>BOGOTÁ / FAC / 2026.05</span>
      </div>
    </section>
  );
}

function Coord({
  label,
  value,
  suffix,
}: {
  label: string;
  value: string;
  suffix: string;
}) {
  return (
    <div className="flex flex-col gap-1 border-l border-rule pl-3">
      <span className="text-ink-faded">{label}</span>
      <span className="font-display font-light text-signal text-2xl normal-case tracking-tight">
        {value}
      </span>
      <span className="text-ink-quiet">{suffix}</span>
    </div>
  );
}
