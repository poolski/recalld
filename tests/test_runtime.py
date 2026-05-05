import asyncio

import pytest

from recalld.runtime import cancel_pipeline_tasks, spawn_pipeline_task


@pytest.mark.asyncio
async def test_cancel_pipeline_tasks_cancels_running_tasks():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker():
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    spawn_pipeline_task(worker())
    await asyncio.wait_for(started.wait(), timeout=1)
    await cancel_pipeline_tasks()
    await asyncio.wait_for(cancelled.wait(), timeout=1)
