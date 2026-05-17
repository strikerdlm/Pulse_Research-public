import {
  useExperiment,
  useExperimentData,
  useRunExperiment,
} from "../hooks/useExperiments";
import { useExperimentEvents } from "../hooks/useExperimentEvents";
import { DesignSpacePlot } from "./DesignSpacePlot";
import { HairlineRule } from "./HairlineRule";
import { ShapBarPanel } from "./ShapBarPanel";
import { SobolTornadoPanel } from "./SobolTornadoPanel";
import { ProgressMeter } from "./ProgressMeter";
import { RunnerBadge } from "./RunnerBadge";
import { StatusPill } from "./StatusPill";

interface Props {
  experimentId: string | null;
}

export function ExperimentDetail({ experimentId }: Props) {
  const detail = useExperiment(experimentId);
  const data = useExperimentData(experimentId);
  const event = useExperimentEvents(experimentId);
  const run = useRunExperiment();

  if (!experimentId) {
    return (
      <div className="border border-rule bg-panel p-12 text-center">
        <p className="font-display italic text-2xl text-ink-faded">
          no run selected.
        </p>
        <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.2em] text-ink-quiet">
          select a row above to inspect.
        </p>
      </div>
    );
  }

  if (detail.isLoading || !detail.data) {
    return (
      <div className="border border-rule bg-panel p-12">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faded">
          retrieving…
        </p>
      </div>
    );
  }

  const exp = detail.data;
  const liveProgress = event?.progress ?? exp.progress;
  const liveStatus = event?.status ?? exp.status;
  const canRun = liveStatus === "pending";

  const variant =
    liveStatus === "completed"
      ? "done"
      : liveStatus === "failed"
        ? "fail"
        : liveStatus === "running"
          ? "running"
          : "idle";

  return (
    <article className="border border-rule bg-panel">
      <header className="px-6 py-4 flex items-baseline justify-between border-b border-rule">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faded">
            exp / {exp.id.slice(0, 8)}
          </p>
          <h2
            className="font-display font-light text-2xl text-ink mt-1"
            style={{ fontVariationSettings: '"opsz" 72, "SOFT" 30' }}
          >
            {exp.name}
          </h2>
        </div>
        <div className="flex flex-col items-end gap-2">
          <StatusPill status={liveStatus} />
          <div className="flex items-center gap-2">
            {exp.engine_label && <RunnerBadge kind={exp.engine_label} />}
            {exp.failed_rows > 0 && (
              <span
                data-testid="failed-rows-chip"
                className="inline-flex items-center gap-1.5 border border-warn/40 text-warn px-2 py-[2px] font-mono text-[10px] tracking-[0.18em]"
              >
                <span aria-hidden="true">!</span>
                {exp.failed_rows} FAILED
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="px-6 py-5 grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Field label="n_base" value={exp.n_base.toString()} />
        <Field label="design rows" value={exp.n_design_rows.toString()} />
        <Field label="seed" value={exp.seed.toString()} />
        <Field
          label="created"
          value={new Date(exp.created_at).toISOString().slice(0, 19).replace("T", " ")}
          small
        />
      </div>

      <div className="px-6 pb-4">
        <ProgressMeter value={liveProgress} label="TELEMETRY" variant={variant} />
      </div>

      <div className="px-6 pb-6">
        <button
          type="button"
          disabled={!canRun || run.isPending}
          onClick={() => run.mutate(exp.id)}
          className={`border px-4 py-2 font-mono text-[11px] uppercase tracking-[0.2em] transition-colors ${
            canRun
              ? "border-signal/60 text-signal hover:bg-signal/10"
              : "border-rule text-ink-quiet cursor-not-allowed"
          }`}
        >
          {canRun ? "▸ commence run" : `cannot run · ${liveStatus}`}
        </button>
        {run.isError && (
          <p className="mt-2 font-mono text-[10px] text-warn">
            {(run.error as Error).message}
          </p>
        )}
      </div>

      <HairlineRule label="design envelope" />
      <div className="p-6 pt-0">
        {data.data ? (
          <DesignSpacePlot
            rows={data.data.rows}
            outputs={data.data.outputs}
            axes={data.data.axes}
            outputRange={data.data.output_range}
            subtitle={`${data.data.n_returned} / ${data.data.n_design_rows} rows · seed ${exp.seed}`}
          />
        ) : (
          <div className="bg-panel border border-rule p-12 text-center">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faded">
              loading design…
            </p>
          </div>
        )}
      </div>
      <SobolTornadoPanel
        experimentId={exp.id}
        status={exp.status}
        hasOutputs={exp.has_outputs}
      />
      <ShapBarPanel
        experimentId={exp.id}
        status={exp.status}
        hasOutputs={exp.has_outputs}
      />
    </article>
  );
}

function Field({
  label,
  value,
  small,
}: {
  label: string;
  value: string;
  small?: boolean;
}) {
  return (
    <div className="flex flex-col gap-1 border-l border-rule pl-3">
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faded">
        {label}
      </span>
      <span
        className={`font-mono text-ink tabular-nums ${
          small ? "text-[11px]" : "text-[16px]"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
