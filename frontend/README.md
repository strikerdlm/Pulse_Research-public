# Pulse_Research Frontend

Single-page React 19 + TypeScript dashboard that drives the FastAPI orchestration layer.

## Aesthetic direction (locked 2026-05-14)

Aerospace instrumentation panel × editorial scientific journal. Charcoal-navy void background with a single Mach-meter amber accent; hairline borders; controlled density. Signature element: a thin "Mission Control" status strip at the top showing live UTC, run counts, active runs, and a pulsing dot indicating API health.

| Layer | Choice | Why this one |
| --- | --- | --- |
| Display type | **Newsreader** (variable serif; opsz 6–72, SOFT, WONK axes) | Editorial weight; the variable axes give us optical sizing and quirky character that avoid generic-AI-dashboard tells. |
| Mono / UI / numerics | **Space Mono** | Distinctive descenders and curls; tabular figures; less common than JetBrains/IBM Plex in 2026 dashboards. |
| Charting | **Apache ECharts 6** via `echarts-for-react` | Same JSON specs feed the publication SVGs through the `echarts` skill — one source of truth from UI to manuscript figure. |
| State | **TanStack Query 5** + native `EventSource` | Server cache + SSE built into the browser; no extra WebSocket / pubsub library for MVP. |
| Build | **Vite 6 + Tailwind v4** (CSS-first `@theme`) | Sub-second HMR; no Next.js / RSC complexity for an internal SPA. |

The token palette is in `src/styles/global.css`:

```css
--color-void: #0a0e16;       /* deep background */
--color-panel: #131722;      /* surface */
--color-rule: #1f2a3a;       /* hairline borders */
--color-ink: #e8e3d6;        /* warm off-white */
--color-signal: #d4a657;     /* Mach amber, primary accent */
--color-trace: #7fb4c9;      /* data trace */
--color-warn: #c44545;       /* NACA red */
--color-spec: #82a87c;       /* in-spec green */
```

## Quickstart

```bash
cd frontend
npm install
npm run dev              # http://localhost:5173, proxies /api → :8000

# In another terminal, start the backend:
cd ..
source .venv/bin/activate
uvicorn pulse_research.api.app:create_app --factory --reload
```

Scripts:

| Script | What it does |
| --- | --- |
| `npm run dev` | Vite dev server with HMR and `/api` proxy. |
| `npm run build` | Type-check (`tsc -b`) then production build to `dist/`. |
| `npm run preview` | Serve the production build locally. |
| `npm run lint` | ESLint 9 flat config. |
| `npm run typecheck` | TypeScript project references (no emit). |
| `npm test` | Vitest 2 with happy-dom + React Testing Library. |

## What is NOT in this MVP

- **Routing.** Single page. TanStack Router lands when there is a second view to navigate to.
- **3D R3F G-LOC envelope.** Deferred; the parallel-coordinates plot covers the MVP visualization need.
- **MSW mock service worker.** Tests use direct fetch mocks where needed.
- **Production deploy mode.** Currently `npm run preview` for local; bundling alongside the FastAPI app comes with Phase 4.5's Docker compose.

## Anti-patterns to avoid (carried forward from research)

- Recharts > 5k points (jank).
- Plotly.js as the only chart lib (~3 MB bundle).
- Vega-Lite as primary (JSON verbosity in TS).
- Material UI / Tremor / Chakra defaults (looks AI-generated).
- Next.js App Router for this internal SPA (RSC fights three.js/Plotly/ECharts).
- WebSockets for one-way status (SSE handles it lighter).
- Storing server data in Zustand/Redux (duplicate the truth).
- Light mode default.
