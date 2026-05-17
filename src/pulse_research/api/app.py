"""FastAPI application factory.

Wire-up: ``InMemoryStore`` + ``EventBroker`` + ``SyntheticRunner`` live on
``app.state``; routers reach them through ``request.app.state.*``. This
pattern keeps the dependency graph explicit and makes test-time substitution
(e.g. a faster batch_size or a mocked runner) trivial via factory kwargs.
"""
from __future__ import annotations

from fastapi import FastAPI

from pulse_research.api.events import EventBroker
from pulse_research.api.routers import experiments, meta, sse
from pulse_research.api.runners import (
    Runner,
    make_runner,
    resolve_runner_kind,
)
from pulse_research.api.store import InMemoryStore


def create_app(
    *,
    store: InMemoryStore | None = None,
    broker: EventBroker | None = None,
    runner: Runner | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    All three collaborators (store, broker, runner) can be injected for tests.
    When ``runner`` is ``None`` we read :data:`PULSE_RESEARCH_RUNNER` from the
    environment (defaulting to ``"synthetic"``) and construct the matching
    runner. Misconfig fails loudly at startup rather than silently degrading.
    """
    app = FastAPI(
        title="Pulse_Research API",
        version="0.1.0",
        description=(
            "Experiment orchestration for the CGEM-Pulse multi-fidelity "
            "G-LOC tolerance surrogate."
        ),
    )

    store = store if store is not None else InMemoryStore()
    broker = broker if broker is not None else EventBroker()
    if runner is None:
        kind = resolve_runner_kind()
        runner = make_runner(kind, store, broker)

    app.state.store = store
    app.state.broker = broker
    app.state.runner = runner

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": app.version}

    app.include_router(experiments.router)
    app.include_router(sse.router)
    app.include_router(meta.router)
    return app
