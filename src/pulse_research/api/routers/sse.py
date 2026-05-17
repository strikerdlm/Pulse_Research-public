"""Server-Sent Events stream for experiment status updates.

Each SSE connection subscribes to the broker, emits the current status
immediately, then streams subsequent status events. The stream auto-closes
when the experiment reaches a terminal state (``completed`` or ``failed``).
A 30-second heartbeat keeps proxies from dropping idle connections.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/experiments", tags=["sse"])

HEARTBEAT_SECONDS = 30.0
_TERMINAL = ("completed", "failed")


@router.get("/{experiment_id}/events")
async def stream_events(
    experiment_id: UUID,
    request: Request,
) -> EventSourceResponse:
    store = request.app.state.store
    broker = request.app.state.broker
    exp = store.get(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail="experiment not found")

    queue = broker.subscribe(experiment_id)

    async def generator() -> AsyncIterator[dict[str, Any]]:
        try:
            current_exp = store.get(experiment_id)
            assert current_exp is not None  # checked above
            initial = {
                "status": current_exp.status.value,
                "progress": current_exp.progress,
            }
            yield {"event": "status", "data": json.dumps(initial)}
            if current_exp.status.value in _TERMINAL:
                return

            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=HEARTBEAT_SECONDS,
                    )
                except TimeoutError:
                    yield {"event": "heartbeat", "data": "{}"}
                    continue
                yield {"event": "status", "data": json.dumps(event)}
                if event.get("status") in _TERMINAL:
                    return
        finally:
            broker.unsubscribe(experiment_id, queue)

    return EventSourceResponse(generator())
