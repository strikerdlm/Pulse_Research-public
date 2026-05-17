"""Tests for the async event broker."""
from __future__ import annotations

import asyncio
from uuid import uuid4

from pulse_research.api.events import QUEUE_MAXSIZE, EventBroker


async def test_publish_to_single_subscriber() -> None:
    broker = EventBroker()
    eid = uuid4()
    q = broker.subscribe(eid)
    await broker.publish(eid, {"status": "running", "progress": 0.5})
    event = await asyncio.wait_for(q.get(), timeout=1.0)
    assert event == {"status": "running", "progress": 0.5}


async def test_publish_to_multiple_subscribers() -> None:
    broker = EventBroker()
    eid = uuid4()
    q1 = broker.subscribe(eid)
    q2 = broker.subscribe(eid)
    await broker.publish(eid, {"status": "completed", "progress": 1.0})
    e1 = await asyncio.wait_for(q1.get(), timeout=1.0)
    e2 = await asyncio.wait_for(q2.get(), timeout=1.0)
    assert e1 == e2 == {"status": "completed", "progress": 1.0}


async def test_unsubscribed_subscriber_does_not_receive() -> None:
    broker = EventBroker()
    eid = uuid4()
    q1 = broker.subscribe(eid)
    q2 = broker.subscribe(eid)
    broker.unsubscribe(eid, q1)
    await broker.publish(eid, {"status": "running", "progress": 0.1})
    assert q1.empty()
    assert not q2.empty()


async def test_subscriber_count() -> None:
    broker = EventBroker()
    eid = uuid4()
    assert broker.subscriber_count(eid) == 0
    q = broker.subscribe(eid)
    assert broker.subscriber_count(eid) == 1
    broker.unsubscribe(eid, q)
    assert broker.subscriber_count(eid) == 0


async def test_full_queue_drops_oldest() -> None:
    broker = EventBroker()
    eid = uuid4()
    q = broker.subscribe(eid)
    for i in range(QUEUE_MAXSIZE + 5):
        await broker.publish(eid, {"i": i})
    # Queue cap holds; we keep the most-recent events (drop-oldest policy).
    assert q.qsize() == QUEUE_MAXSIZE
    first = await q.get()
    # The oldest five publications were dropped, so the first remaining
    # element should be the 5th publication (i=5).
    assert first["i"] == 5
