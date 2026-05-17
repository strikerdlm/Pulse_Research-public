import { useEffect, useState } from "react";

import { useExperiments, useHealth } from "../hooks/useExperiments";
import { useRunnerInfo } from "../hooks/useRunnerInfo";
import { RunnerBadge } from "./RunnerBadge";

function pad(n: number, w = 4): string {
  return String(n).padStart(w, "0");
}

function fmtClock(d: Date): string {
  return [d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds()]
    .map((n) => String(n).padStart(2, "0"))
    .join(":");
}

/**
 * Top status strip — the signature element. Reads experiment data + API
 * health and renders Mission-Control-style telemetry: live UTC clock, run
 * counts, active runs, last sync, with a single pulsing dot indicating
 * connection state.
 */
export function MissionControl() {
  const exps = useExperiments();
  const health = useHealth();
  const runnerInfo = useRunnerInfo();

  const [clock, setClock] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const total = exps.data?.length ?? 0;
  const running = exps.data?.filter((e) => e.status === "running").length ?? 0;
  const completed = exps.data?.filter((e) => e.status === "completed").length ?? 0;

  const ok = health.isSuccess;
  const activeKind = runnerInfo.data?.active_kind;

  return (
    <header
      data-testid="mission-control"
      className="border-b border-rule bg-panel/60 backdrop-blur"
    >
      <div className="px-6 py-2 flex items-center gap-8 text-[10px] uppercase tracking-[0.2em] font-mono text-ink-faded">
        <span className="flex items-center gap-2">
          <span className={`signal-dot ${ok ? "" : "signal-dot-warn"}`} />
          <span className="text-ink">
            {ok ? "LINK / NOMINAL" : "LINK / SEEKING"}
          </span>
        </span>

        <span className="text-rule-bright">/</span>

        <Field label="UTC"      value={fmtClock(clock)} />
        <Field label="RUNS"     value={pad(total)}      />
        <Field label="ACTIVE"   value={pad(running, 2)} accent={running > 0} />
        <Field label="DONE"     value={pad(completed)}  />
        <Field label="API"      value={health.data?.version ?? "—"} />

        <span className="ml-auto flex items-center gap-3 text-ink-faded">
          <span>MISSION_CTRL // CGEM-PULSE / hypoxia surrogate</span>
          {activeKind && <RunnerBadge kind={activeKind} />}
        </span>
      </div>
    </header>
  );
}

interface FieldProps {
  label: string;
  value: string;
  accent?: boolean;
}

function Field({ label, value, accent }: FieldProps) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span>{label}</span>
      <span className={`text-ink ${accent ? "text-signal" : ""}`}>{value}</span>
    </span>
  );
}
