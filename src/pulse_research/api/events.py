"""Async per-experiment pub/sub broker for Server-Sent Events.

Each SSE subscriber gets its own ``asyncio.Queue`` registered against the
experiment ID. The runner publishes status events to the broker; the broker
fan-outs to all queues. Bounded queue size prevents memory blow-up if a
subscriber stalls.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from typing import Any
from uuid import UUID

QUEUE_MAXSIZE = 256


class EventBroker:
    """Per-experiment asyncio.Queue pub/sub."""

    def __init__(self) -> None:
        self._subscribers: dict[UUID, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)

    async def publish(
        self,
        experiment_id: UUID,
        event: dict[str, Any],
    ) -> None:
        for q in list(self._subscribers.get(experiment_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest, push the new one — preserves latest state.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                q.put_nowait(event)

    def subscribe(self, experiment_id: UUID) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._subscribers[experiment_id].append(q)
        return q

    def unsubscribe(
        self,
        experiment_id: UUID,
        q: asyncio.Queue[dict[str, Any]],
    ) -> None:
        if q in self._subscribers.get(experiment_id, []):
            self._subscribers[experiment_id].remove(q)

    def subscriber_count(self, experiment_id: UUID) -> int:
        return len(self._subscribers.get(experiment_id, []))
