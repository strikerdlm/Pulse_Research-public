import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";

/**
 * Parallel-coordinates plot over the 11-axis Sobol design space. The two
 * hypoxia axes (fio2_inspired, sao2_baseline) carry an amber accent on
 * their axis labels. When ``outputs`` contains a populated range, lines are
 * coloured by output value via an ECharts ``visualMap`` over a viridis-style
 * ramp; otherwise lines render in a single cool-grey trace tone.
 *
 * Pure render — data comes from the parent. The Phase 5 placeholder
 * mulberry32 generator has been removed.
 */

const ACCENT_AXES = new Set(["fio2_inspired", "sao2_baseline"]);

interface Props {
  rows: number[][];
  outputs: (number | null)[];
  axes: string[];
  outputRange: { min: number | null; max: number | null };
  subtitle?: string;
}

export function DesignSpacePlot({
  rows,
  outputs,
  axes,
  outputRange,
  subtitle,
}: Props) {
  const hasOutputs =
    outputRange.min !== null &&
    outputRange.max !== null &&
    outputRange.min !== outputRange.max;

  // Append the output as the last column of each row so visualMap can read it.
  const lineColumns = axes.length;
  const data: number[][] = rows.map((row, i) => {
    const out = outputs[i];
    return [...row, out === null ? Number.NaN : out];
  });

  const parallelAxes = axes.map((name, i) => ({
    dim: i,
    name,
    nameTextStyle: {
      color: ACCENT_AXES.has(name) ? "#d4a657" : "#6b7283",
      fontFamily: "Space Mono, monospace",
      fontSize: ACCENT_AXES.has(name) ? 10 : 9,
    },
  }));

  const option: EChartsOption = {
    backgroundColor: "transparent",
    color: ["#7fb4c9"],
    ...(hasOutputs && {
      visualMap: {
        type: "continuous",
        min: outputRange.min as number,
        max: outputRange.max as number,
        dimension: lineColumns,
        inRange: {
          color: ["#440154", "#3b528b", "#21908d", "#5dc863", "#fde725"],
        },
        textStyle: {
          color: "#6b7283",
          fontFamily: "Space Mono, monospace",
          fontSize: 9,
        },
        left: 20,
        bottom: 20,
        itemHeight: 100,
        itemWidth: 8,
        precision: 1,
      },
    }),
    parallel: {
      left: hasOutputs ? "12%" : "8%",
      right: "8%",
      top: "12%",
      bottom: "16%",
      parallelAxisDefault: {
        type: "value",
        nameLocation: "end",
        nameRotate: 30,
        nameGap: 14,
        nameTextStyle: {
          color: "#6b7283",
          fontFamily: "Space Mono, monospace",
          fontSize: 9,
          fontWeight: 400,
        },
        axisLine: { lineStyle: { color: "#1f2a3a", width: 1 } },
        axisTick: { lineStyle: { color: "#2f3d54" } },
        axisLabel: {
          color: "#3a4252",
          fontFamily: "Space Mono, monospace",
          fontSize: 9,
        },
        splitLine: { show: false },
      },
    },
    parallelAxis: parallelAxes,
    series: [
      {
        type: "parallel",
        smooth: false,
        lineStyle: {
          width: 1,
          color: "#7fb4c9",
          opacity: hasOutputs ? 0.45 : 0.18,
        },
        data,
      },
    ],
  };

  return (
    <div className="bg-panel border border-rule" data-testid="design-space-plot">
      <div className="px-4 pt-3 pb-1 border-b border-rule flex items-baseline justify-between">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faded">
          design space // {axes.length} axes (sobol)
        </p>
        <p className="font-mono text-[10px] text-ink-quiet">
          {subtitle ?? `${rows.length} rows`}
        </p>
      </div>
      <ReactECharts
        option={option}
        style={{ height: 320, width: "100%" }}
        theme="dark"
        opts={{ renderer: "svg" }}
        notMerge
      />
    </div>
  );
}
