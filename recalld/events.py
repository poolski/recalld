from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import AsyncIterator


class EventBus:
    """Per-job SSE event bus. Consumers subscribe to a job_id and receive events as JSON strings."""

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def publish(self, job_id: str, event: dict) -> None:
        import json
        data = json.dumps(event)
        for q in self._queues[job_id]:
            q.put_nowait(data)

    async def subscribe(self, job_id: str) -> AsyncIterator[str]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues[job_id].append(q)
        try:
            while True:
                data = await q.get()
                yield data
                if data == '"done"':
                    break
        finally:
            self._queues[job_id].remove(q)


bus = EventBus()
