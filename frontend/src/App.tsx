import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { CreateExperimentForm } from "./components/CreateExperimentForm";
import { ExperimentDetail } from "./components/ExperimentDetail";
import { ExperimentsTable } from "./components/ExperimentsTable";
import { Hero } from "./components/Hero";
import { HairlineRule } from "./components/HairlineRule";
import { MissionControl } from "./components/MissionControl";
import { ValidationLab } from "./components/ValidationLab";
import { useExperiments } from "./hooks/useExperiments";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1 },
  },
});

function Console() {
  const exps = useExperiments();
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <main className="min-h-screen bg-void text-ink">
      <MissionControl />
      <Hero />
      <ValidationLab />

      <div className="px-6 py-8 max-w-[1400px] grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6">
        <div>
          <HairlineRule label="register" />
          <ExperimentsTable
            experiments={exps.data ?? []}
            selectedId={selected}
            onSelect={setSelected}
          />

          <div className="mt-8">
            <HairlineRule label="selected run" />
            <ExperimentDetail experimentId={selected} />
          </div>
        </div>

        <aside className="space-y-6 lg:sticky lg:top-12 lg:self-start">
          <section className="border border-rule bg-panel p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.22em] text-ink-faded mb-4">
              new experiment
            </p>
            <CreateExperimentForm onCreated={setSelected} />
          </section>

          <section className="border border-rule bg-panel-2 p-6 text-[11px] text-ink-faded leading-relaxed">
            <p className="font-mono uppercase tracking-[0.22em] text-ink mb-3">
              method
            </p>
            <p className="font-display">
              Each experiment generates an{" "}
              <span className="text-signal">11-axis Saltelli-Sobol design</span>{" "}
              over the CGEM-Pulse feature space. The CGEM acceleration arm is
              hypoxia-blind by design; the Pulse hypoxia arm is driven by an
              FiO₂-threshold environment tier (Standard / Hypobaric-3000 m /
              Hypobaric-4000 m). The continuous FiO₂ dependence in the corrected
              tolerance time enters analytically through the{" "}
              <span className="text-signal">Hüfner CaO₂ ratio</span>, with no
              fitted coupling parameter. Progress is streamed over Server-Sent
              Events.
            </p>
          </section>
        </aside>
      </div>

      <footer className="border-t border-rule px-6 py-4 mt-8 font-mono text-[10px] uppercase tracking-[0.22em] text-ink-quiet flex justify-between">
        <span>PULSE_RESEARCH // v0.1.0 // MIT</span>
        <span>strikerdlm / Pulse_Research</span>
      </footer>
    </main>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Console />
    </QueryClientProvider>
  );
}
