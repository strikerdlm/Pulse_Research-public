/**
 * Validation Lab — Phase 7.3 snapshot showcase.
 *
 * A magazine-style multi-panel section that visualises the manuscript's
 * headline results in the aerospace-instrumentation aesthetic. Reads from
 * the static `phase73Snapshot` data module (canonical numbers, frozen at
 * the v0.7.3-paper-submission tag); no backend dependency.
 *
 * Panels:
 *   1. Hüfner coupling — equation rendered as editorial typography.
 *   2. Whinnery envelope — ECharts scatter, 8 in-scope bins vs WF2013 means,
 *      with within-bin distribution-envelope band.
 *   3. Coupling-layer ablation — before/after S₁[FiO₂] contrast.
 *   4. Conformal coverage — Mondrian altitude-tier bar chart.
 */
import { useMemo } from "react";
import ReactECharts from "echarts-for-react";

import {
  ABLATION,
  CONFORMAL,
  DIRECTIONAL,
  HUFNER,
  PHASE_TAG,
  WHINNERY,
} from "../data/phase73Snapshot";

const SIGNAL = "#d4a657";
const TRACE = "#7fb4c9";
const SPEC = "#82a87c";
const WARN = "#c44545";
const RULE = "#1f2a3a";
const INK_FADED = "#6b7283";
const INK_QUIET = "#3a4252";

export function ValidationLab() {
  return (
    <section
      id="validation-lab"
      aria-labelledby="validation-lab-heading"
      className="relative border-t border-rule"
    >
      {/* Section header — small-caps eyebrow + display title + decorative coord band */}
      <div className="px-6 pt-12 pb-4 max-w-[1400px]">
        <div className="flex items-baseline justify-between mb-6">
          <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-faded">
            §3 · phase 7.3 validation snapshot
          </p>
          <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-ink-quiet">
            release {PHASE_TAG.release} · {PHASE_TAG.commit_short}
          </p>
        </div>

        <h2
          id="validation-lab-heading"
          className="font-display font-light text-ink tracking-[-0.015em]"
          style={{
            fontSize: "clamp(1.75rem, 3.6vw, 3.25rem)",
            lineHeight: 1.02,
            fontVariationSettings: '"opsz" 60, "SOFT" 30, "WONK" 0',
          }}
        >
          <span className="text-signal italic">Calibrated</span> against{" "}
          <span className="italic">{WHINNERY.corpus_total}</span> centrifuge episodes;
          <br />
          coupled by a closed-form ratio with{" "}
          <span className="text-signal italic">no fitted parameter</span>.
        </h2>

        <p className="mt-6 max-w-[58ch] font-display text-[13px] leading-[1.65] text-ink-faded">
          Four diagnostics drawn from the frozen Phase 7.3 export bundle:
          the Hüfner coupling layer that introduces the hypoxia channel
          analytically; the Whinnery & Forster (2013) within-bin
          distribution-envelope plausibility check; the coupling-layer
          ablation that isolates the FiO₂ variance contribution; and the
          Mondrian-stratified split-conformal coverage diagnostic over the
          deterministic composite output.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-px bg-rule border-t border-b border-rule">
        {/* Panel 1 — Hüfner coupling (typographic) */}
        <HufnerPanel />

        {/* Panel 2 — Whinnery envelope chart */}
        <WhinneryPanel />

        {/* Panel 3 — Coupling ablation contrast */}
        <AblationPanel />

        {/* Panel 4 — Conformal coverage Mondrian strata */}
        <ConformalPanel />
      </div>

      {/* Bottom coordinate band — directional check */}
      <DirectionalStrip />
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────── */

function HufnerPanel() {
  return (
    <article className="bg-panel p-6 lg:col-span-5 grain relative overflow-hidden">
      <header className="flex items-baseline justify-between mb-5">
        <PanelLabel
          ordinal="01"
          title="Hüfner coupling"
          subtitle="no-fitted-parameter layer"
        />
        <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-quiet">
          §2.3
        </span>
      </header>

      <div className="my-10 flex flex-col items-center text-center">
        {/* The corrected time as a typographically composed identity */}
        <div
          className="font-display text-ink flex items-center gap-4"
          style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30, "WONK" 0' }}
        >
          <span className="italic text-signal text-[44px] leading-none">t̂</span>
          <span className="text-ink-quiet text-[28px]">=</span>
          <span className="italic text-[34px] leading-none">
            t<sub className="text-[12px] not-italic text-ink-faded tracking-[0.1em]">CGEM</sub>
          </span>
          <span className="text-ink-quiet text-[28px]">·</span>
          <span className="inline-flex flex-col items-center text-[18px]">
            <span className="border-b border-signal px-4 pb-1.5 text-ink whitespace-nowrap">
              C<sub className="text-[11px] text-ink-faded">aO₂</sub>
              <span className="text-ink-quiet mx-0.5">(</span>
              <span className="italic">S</span><sub className="text-[11px] not-italic text-ink-faded">aO₂</sub>
              <span className="text-ink-quiet">, </span>
              <span className="italic">P</span><sub className="text-[11px] not-italic text-ink-faded">aO₂</sub>
              <span className="text-ink-quiet">)</span>
            </span>
            <span className="px-4 pt-1.5 text-ink-faded whitespace-nowrap">
              C<sub className="text-[11px]">aO₂, baseline</sub>
            </span>
          </span>
        </div>

        <div className="mt-7 font-display text-[14px] text-ink leading-[1.8]">
          <span className="italic">C</span><sub className="text-[10px] not-italic text-ink-faded">aO₂</sub>
          <span className="text-ink-quiet mx-2">=</span>
          <span className="text-signal font-mono text-[13px]">{HUFNER.hufner_mL_per_g}</span>
          <span className="text-ink-quiet mx-1.5">·</span>
          <span className="italic">Hb</span>
          <span className="text-ink-quiet mx-1.5">·</span>
          <span className="italic">S</span><sub className="text-[10px] not-italic text-ink-faded">aO₂</sub>
          <span className="text-ink-quiet mx-2">+</span>
          <span className="text-signal font-mono text-[13px]">{HUFNER.henry_mL_per_dL_per_mmHg}</span>
          <span className="text-ink-quiet mx-1.5">·</span>
          <span className="italic">P</span><sub className="text-[10px] not-italic text-ink-faded">aO₂</sub>
        </div>
      </div>

      <dl className="mt-6 grid grid-cols-3 gap-x-3 gap-y-2 text-[9px] uppercase tracking-[0.18em] font-mono">
        <ConstantCell label="Hb" value={`${HUFNER.hb_g_per_dL}`} unit="g/dL" />
        <ConstantCell label="PaCO₂" value={`${HUFNER.paco2_mmHg}`} unit="mmHg" />
        <ConstantCell label="RQ" value={`${HUFNER.rq}`} unit="—" />
        <ConstantCell label="Hüfner" value={`${HUFNER.hufner_mL_per_g}`} unit="mL O₂/g" />
        <ConstantCell label="Henry" value={`${HUFNER.henry_mL_per_dL_per_mmHg}`} unit="mL/dL/mmHg" />
        <ConstantCell label="altitude" value={`${HUFNER.altitude_m}`} unit="m (sea-level)" />
      </dl>

      <p className="mt-5 font-mono text-[10px] uppercase tracking-[0.22em] text-ink-quiet leading-[1.6]">
        Six constants. None tuned. All read from canonical clinical references
        before any contact with the validation anchor.
      </p>
    </article>
  );
}

function ConstantCell({ label, value, unit }: { label: string; value: string; unit: string }) {
  return (
    <div className="flex flex-col gap-0.5 border-l border-rule pl-2.5">
      <span className="text-ink-quiet">{label}</span>
      <span className="font-display normal-case tracking-tight text-signal text-[15px]">{value}</span>
      <span className="text-ink-quiet text-[8px]">{unit}</span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────── */

function WhinneryPanel() {
  const option = useMemo(() => buildWhinneryOption(), []);

  return (
    <article className="bg-panel p-6 lg:col-span-7 relative">
      <header className="flex items-baseline justify-between mb-4">
        <PanelLabel
          ordinal="02"
          title="Whinnery envelope"
          subtitle="bin-mean plausibility · n=8 in-scope ROR"
        />
        <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-quiet">
          §3.5
        </span>
      </header>

      <div className="grid grid-cols-3 gap-3 mb-4">
        <Metric
          big={`${(WHINNERY.bin_fraction_in_envelope * 100).toFixed(0)} %`}
          small="in envelope"
          accent
        />
        <Metric
          big={WHINNERY.bin_median_abs_z.toFixed(3)}
          small="median |z|"
        />
        <Metric
          big={`${WHINNERY.closed_form_mae_s.toFixed(2)} s`}
          small={`MAE · n=${WHINNERY.closed_form_n}`}
        />
      </div>

      <ReactECharts option={option} style={{ height: 320 }} opts={{ renderer: "svg" }} />

      <p className="mt-3 font-mono text-[9.5px] uppercase tracking-[0.18em] text-ink-quiet leading-[1.7]">
        Within-bin distribution-envelope plausibility — not mean-level
        validation. Identity line and shaded ±1.96σ band are orientation aids;
        the pre-specified diagnostic is the per-bin |z| against the paper's
        within-bin standard deviation.
      </p>
    </article>
  );
}

function buildWhinneryOption() {
  // Build the scatter: x = paper bin mean, y = surrogate predicted mean.
  // Marker size encodes n_paper; marker fill darkens with z.
  const scatterData = WHINNERY.bins.map((b) => ({
    name: b.label,
    value: [b.paper_mean, b.predicted, b.n_paper, b.z, b.label, b.table],
    symbolSize: 8 + Math.sqrt(b.n_paper) * 1.6,
    itemStyle: {
      color: b.table === 1 ? TRACE : SIGNAL,
      borderColor: "#0a0e16",
      borderWidth: 1.5,
      opacity: 0.92,
    },
  }));

  const lo = 8.0;
  const hi = 12.0;
  const identityLine = [
    [lo, lo],
    [hi, hi],
  ];

  // Build the y = x ± 1.96·σ envelope band as a markArea
  // Use median per-bin SD ≈ 3.0 s as a visual aid (range 2.15–4.59 s)
  const sd = 3.0;

  return {
    backgroundColor: "transparent",
    grid: { left: 60, right: 30, top: 24, bottom: 48 },
    tooltip: {
      trigger: "item",
      backgroundColor: "#0a0e16",
      borderColor: RULE,
      borderWidth: 1,
      textStyle: { color: "#e8e3d6", fontFamily: "Space Mono, monospace", fontSize: 11 },
      formatter: (p: { value: [number, number, number, number, string, number] }) => {
        const [paper, pred, n, z, lbl, tbl] = p.value;
        return [
          `<span style="color:${SIGNAL}; letter-spacing: 0.12em; text-transform: uppercase">${lbl}</span> · table ${tbl}`,
          `<br/>paper mean: <b>${paper.toFixed(2)} s</b> · n=${n}`,
          `<br/>predicted:  <b>${pred.toFixed(3)} s</b>`,
          `<br/>z-score:    <b style="color: ${Math.abs(z) < 1.96 ? SPEC : WARN}">${z.toFixed(3)}</b>`,
        ].join("");
      },
    },
    xAxis: {
      type: "value",
      name: "paper bin-mean (s)",
      nameLocation: "middle",
      nameGap: 30,
      nameTextStyle: { color: INK_FADED, fontFamily: "Space Mono, monospace", fontSize: 10 },
      min: lo,
      max: hi,
      splitNumber: 4,
      axisLine: { lineStyle: { color: RULE } },
      axisTick: { lineStyle: { color: RULE } },
      axisLabel: { color: INK_FADED, fontFamily: "Space Mono, monospace", fontSize: 10 },
      splitLine: { lineStyle: { color: INK_QUIET, type: "dashed", opacity: 0.4 } },
    },
    yAxis: {
      type: "value",
      name: "surrogate (s)",
      nameLocation: "middle",
      nameGap: 42,
      nameTextStyle: { color: INK_FADED, fontFamily: "Space Mono, monospace", fontSize: 10 },
      min: lo,
      max: hi,
      splitNumber: 4,
      axisLine: { lineStyle: { color: RULE } },
      axisTick: { lineStyle: { color: RULE } },
      axisLabel: { color: INK_FADED, fontFamily: "Space Mono, monospace", fontSize: 10 },
      splitLine: { lineStyle: { color: INK_QUIET, type: "dashed", opacity: 0.4 } },
    },
    series: [
      {
        type: "custom",
        name: "envelope",
        renderItem: (_params: unknown, api: { coord: (a: [number, number]) => [number, number]; getWidth: () => number }) => {
          const p1 = api.coord([lo, lo - 1.96 * sd]);
          const p2 = api.coord([hi, hi - 1.96 * sd]);
          const p3 = api.coord([hi, hi + 1.96 * sd]);
          const p4 = api.coord([lo, lo + 1.96 * sd]);
          return {
            type: "polygon",
            shape: { points: [p1, p2, p3, p4] },
            style: { fill: TRACE, opacity: 0.06, stroke: "transparent" },
          };
        },
        data: [[0]],
        silent: true,
        z: 1,
      },
      {
        name: "identity",
        type: "line",
        data: identityLine,
        showSymbol: false,
        lineStyle: { color: INK_FADED, type: "dashed", width: 1, opacity: 0.6 },
        silent: true,
        z: 2,
      },
      {
        name: "bins",
        type: "scatter",
        data: scatterData,
        z: 3,
      },
    ],
  };
}

/* ─────────────────────────────────────────────────────────────────────── */

function AblationPanel() {
  return (
    <article className="bg-panel p-6 lg:col-span-5">
      <header className="flex items-baseline justify-between mb-4">
        <PanelLabel
          ordinal="03"
          title="Coupling-layer ablation"
          subtitle="isolates the Hüfner variance contribution"
        />
        <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-quiet">
          §3.3
        </span>
      </header>

      <div className="flex items-center justify-between gap-4 my-6">
        <AblationCell
          label="ablated"
          equation="CaO₂ ratio → 1"
          value={ABLATION.ablated_S1_fio2}
          conf={ABLATION.ablated_S1_fio2_conf}
          ghostly
        />
        <DeltaArrow value={ABLATION.delta} />
        <AblationCell
          label="coupled"
          equation="closed-form Hüfner"
          value={ABLATION.coupled_S1_fio2}
          conf={0.016}
        />
      </div>

      <div className="border border-rule p-4 mt-6 relative">
        <p className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-faded mb-2">
          interpretation
        </p>
        <p className="font-display text-[12px] text-ink leading-[1.55]">
          The CGEM feature translator drops both oxygen axes by construction
          (§2.4). Replacing the C<sub>aO₂</sub> ratio with a constant collapses
          S₁[FiO₂] to the bootstrap floor, attributing the full{" "}
          <span className="text-signal font-mono">+0.249</span> variance share
          to the closed-form coupling layer alone.
        </p>
      </div>
    </article>
  );
}

function AblationCell({
  label,
  equation,
  value,
  conf,
  ghostly,
}: {
  label: string;
  equation: string;
  value: number;
  conf: number;
  ghostly?: boolean;
}) {
  // Render very small numbers in scientific notation
  const display =
    Math.abs(value) < 1e-3 ? value.toExponential(1) : value.toFixed(4);
  const confDisplay = conf < 1e-3 ? conf.toExponential(1) : conf.toFixed(3);

  return (
    <div className="flex-1 flex flex-col items-center text-center">
      <span
        className={`font-mono text-[10px] uppercase tracking-[0.22em] mb-1 ${
          ghostly ? "text-ink-quiet" : "text-ink-faded"
        }`}
      >
        {label}
      </span>
      <span
        className={`font-display normal-case tracking-tight text-[40px] leading-none ${
          ghostly ? "text-ink-quiet" : "text-signal"
        }`}
        style={{ fontVariationSettings: '"opsz" 60, "SOFT" 30' }}
      >
        {display}
      </span>
      <span className="font-mono text-[9px] tracking-[0.12em] text-ink-quiet mt-1">
        ± {confDisplay}
      </span>
      <span
        className={`font-mono text-[9px] uppercase tracking-[0.18em] mt-2 ${
          ghostly ? "text-ink-quiet" : "text-ink-faded"
        }`}
      >
        {equation}
      </span>
    </div>
  );
}

function DeltaArrow({ value }: { value: number }) {
  return (
    <div className="flex flex-col items-center text-center px-2">
      <span className="text-signal font-display text-[20px] leading-none">→</span>
      <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-faded mt-1">
        Δ S₁
      </span>
      <span className="font-display text-signal text-[18px] tracking-tight">
        +{value.toFixed(3)}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────── */

function ConformalPanel() {
  const option = useMemo(() => buildConformalOption(), []);

  return (
    <article className="bg-panel p-6 lg:col-span-7">
      <header className="flex items-baseline justify-between mb-4">
        <PanelLabel
          ordinal="04"
          title="Split-conformal coverage"
          subtitle={`α = ${CONFORMAL.alpha} · Mondrian by altitude tier`}
        />
        <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-quiet">
          §3.4
        </span>
      </header>

      <div className="grid grid-cols-4 gap-3 mb-4">
        <Metric
          big={CONFORMAL.marginal_coverage.toFixed(4)}
          small={`vs. ${CONFORMAL.target.toFixed(2)} target`}
          accent
        />
        <Metric
          big={`${(CONFORMAL.q_hat_s * 1000).toFixed(1)} ms`}
          small="q̂ (half-width)"
        />
        <Metric
          big={`${CONFORMAL.n_test}`}
          small="n_test"
        />
        <Metric
          big={CONFORMAL.mc_propagation.toFixed(4)}
          small="MC sigma"
        />
      </div>

      <ReactECharts option={option} style={{ height: 220 }} opts={{ renderer: "svg" }} />

      <p className="mt-3 font-mono text-[9.5px] uppercase tracking-[0.18em] text-ink-quiet leading-[1.7]">
        Coverage is over the surrogate's reproduction of the deterministic
        CGEM × Hüfner composite, not over real-pilot variability. The
        4 000 m stratum (n=2) is statistically uninformative and is shown for
        completeness only.
      </p>
    </article>
  );
}

function buildConformalOption() {
  const labels = CONFORMAL.strata.map((s) => `${s.label}  (n=${s.n})`);
  const coverages = CONFORMAL.strata.map((s) => +s.coverage.toFixed(4));
  const isDegenerate = CONFORMAL.strata.map((s) => s.n < 10);

  return {
    backgroundColor: "transparent",
    grid: { left: 180, right: 60, top: 12, bottom: 36 },
    xAxis: {
      type: "value",
      min: 0.8,
      max: 1.05,
      axisLine: { lineStyle: { color: RULE } },
      axisTick: { lineStyle: { color: RULE } },
      axisLabel: {
        color: INK_FADED,
        fontFamily: "Space Mono, monospace",
        fontSize: 10,
        formatter: (v: number) => v.toFixed(2),
      },
      splitLine: { lineStyle: { color: INK_QUIET, type: "dashed", opacity: 0.3 } },
    },
    yAxis: {
      type: "category",
      data: labels,
      axisLine: { lineStyle: { color: RULE } },
      axisTick: { show: false },
      axisLabel: {
        color: "#e8e3d6",
        fontFamily: "Space Mono, monospace",
        fontSize: 10,
      },
    },
    series: [
      // Background reference bar showing the 0.90 target line position
      {
        type: "bar",
        data: coverages.map((c, i) =>
          isDegenerate[i] ? { value: c, itemStyle: { color: WARN, opacity: 0.7 } } : { value: c, itemStyle: { color: TRACE } },
        ),
        barWidth: 22,
        label: {
          show: true,
          position: "right",
          color: "#e8e3d6",
          fontFamily: "Space Mono, monospace",
          fontSize: 11,
          formatter: (p: { value: number }) => p.value.toFixed(4),
        },
        markLine: {
          symbol: ["none", "none"],
          lineStyle: { color: SIGNAL, type: "dashed", width: 1.5, opacity: 0.8 },
          label: {
            color: SIGNAL,
            fontFamily: "Space Mono, monospace",
            fontSize: 9,
            formatter: "α = 0.10",
          },
          data: [{ xAxis: CONFORMAL.target }],
        },
        z: 2,
      },
    ],
  };
}

/* ─────────────────────────────────────────────────────────────────────── */

function DirectionalStrip() {
  // The hypoxia directional sanity check — a slim narrative band beneath the
  // four panels. Effect-size first, p-value secondary.
  const dMedian = (DIRECTIONAL.median_high_fio2_s - DIRECTIONAL.median_low_fio2_s).toFixed(2);
  return (
    <div className="px-6 py-5 max-w-[1400px]">
      <div className="flex flex-wrap items-baseline gap-x-8 gap-y-3 font-mono text-[10px] uppercase tracking-[0.22em] text-ink-faded">
        <span className="text-ink">
          <span className="text-signal mr-1">↘</span>
          hypoxia directional check
        </span>
        <span>
          median<span className="text-ink-quiet ml-1">low FiO₂ &lt; {DIRECTIONAL.threshold_low}:</span>{" "}
          <span className="text-ink font-display normal-case tracking-tight text-[14px]">
            {DIRECTIONAL.median_low_fio2_s.toFixed(1)} s
          </span>
        </span>
        <span>
          median<span className="text-ink-quiet ml-1">high FiO₂ &gt; {DIRECTIONAL.threshold_high}:</span>{" "}
          <span className="text-ink font-display normal-case tracking-tight text-[14px]">
            {DIRECTIONAL.median_high_fio2_s.toFixed(1)} s
          </span>
        </span>
        <span className="text-ink">
          Δ = <span className="text-signal">−{dMedian} s</span> · shorter under hypoxia
        </span>
        <span className="ml-auto text-ink-quiet">
          Mann–Whitney U one-sided · p &lt; 10⁻⁶ · n_low={DIRECTIONAL.n_low}, n_high={DIRECTIONAL.n_high.toLocaleString()}
        </span>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────── */

function PanelLabel({
  ordinal,
  title,
  subtitle,
}: {
  ordinal: string;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-quiet">
        {ordinal} /
      </span>
      <h3 className="m-0 font-display text-ink text-[15px] tracking-tight">
        {title}
      </h3>
      <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-ink-faded">
        · {subtitle}
      </span>
    </div>
  );
}

function Metric({
  big,
  small,
  accent,
}: {
  big: string;
  small: string;
  accent?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 border-l border-rule pl-3 py-1">
      <span
        className={`font-display normal-case tracking-tight text-[22px] leading-none ${
          accent ? "text-signal" : "text-ink"
        }`}
        style={{ fontVariationSettings: '"opsz" 48, "SOFT" 20' }}
      >
        {big}
      </span>
      <span className="font-mono text-[9px] uppercase tracking-[0.22em] text-ink-faded">
        {small}
      </span>
    </div>
  );
}

