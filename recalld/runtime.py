from __future__ import annotations

import asyncio
from collections.abc import Coroutine


_pipeline_tasks: set[asyncio.Task] = set()


def spawn_pipeline_task(coro: Coroutine | None) -> asyncio.Task | None:
    if coro is None:
        return None
    task = asyncio.create_task(coro)
    if task is None:
        return None
    _pipeline_tasks.add(task)
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(_pipeline_tasks.discard)
    return task


async def cancel_pipeline_tasks() -> None:
    if not _pipeline_tasks:
        return
    tasks = list(_pipeline_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
