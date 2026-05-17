import { useState } from "react";

import { useCreateExperiment } from "../hooks/useExperiments";

const N_BASE_OPTIONS = [4, 8, 16, 32, 64, 128, 256, 512, 1024];

interface Props {
  onCreated?: (id: string) => void;
}

export function CreateExperimentForm({ onCreated }: Props) {
  const [name, setName] = useState("");
  const [nBase, setNBase] = useState(64);
  const [seed, setSeed] = useState(42);
  const create = useCreateExperiment();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    create.mutate(
      { name: name.trim(), n_base: nBase, seed },
      {
        onSuccess: (data) => {
          setName("");
          onCreated?.(data.id);
        },
      },
    );
  };

  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="space-y-1">
        <Label>name</Label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. baseline-21-pct-o2"
          className="w-full bg-void border border-rule px-3 py-2 font-mono text-[12px] text-ink placeholder:text-ink-quiet focus:outline-none focus:border-signal/60"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <Label>n_base (rows = N·24)</Label>
          <select
            value={nBase}
            onChange={(e) => setNBase(Number(e.target.value))}
            className="w-full bg-void border border-rule px-3 py-2 font-mono text-[12px] text-ink focus:outline-none focus:border-signal/60"
          >
            {N_BASE_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n}  →  {n * 24}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1">
          <Label>seed</Label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-full bg-void border border-rule px-3 py-2 font-mono text-[12px] text-ink focus:outline-none focus:border-signal/60"
          />
        </div>
      </div>

      {create.isError && (
        <p className="font-mono text-[11px] text-warn">
          {(create.error as Error).message}
        </p>
      )}

      <button
        type="submit"
        disabled={create.isPending || !name.trim()}
        className="w-full border border-signal/60 text-signal font-mono text-[11px] uppercase tracking-[0.2em] py-2 hover:bg-signal/10 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {create.isPending ? "transmitting…" : "▸ create experiment"}
      </button>
    </form>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label className="block font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faded">
      {children}
    </label>
  );
}
