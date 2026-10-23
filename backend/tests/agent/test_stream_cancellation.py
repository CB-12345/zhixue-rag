import asyncio
import json
from types import SimpleNamespace

import pytest

from app.agent import agent as module


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("closing", [False, True])
async def test_disconnected_stream_stops_model_work_before_releasing_lock(monkeypatch, resume, closing):
    started, cancelled = asyncio.Event(), asyncio.Event()
    tasks, deleted = [], []

    class Manager:
        async def get_pending_run_id(self, *args):
            return "resume-run" if resume else None

        async def get_history(self, *args):
            return []

        async def add_message(self, *args, **kwargs):
            pytest.fail("An interrupted answer must not be saved as a completed message")

    async def emit(*args):
        tasks.append(asyncio.current_task())
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def delete(run_id):
        deleted.append(run_id)

    monkeypatch.setattr(module, "sm", SimpleNamespace(session_manager=Manager()))
    monkeypatch.setattr(module.agent_factory, "create_agent", lambda **kwargs: object())
    monkeypatch.setattr(module, "_emit_agent_events", emit)
    monkeypatch.setattr(module, "get_checkpointer", lambda: SimpleNamespace(adelete_thread=delete))
    stream = (module.get_agent_resume_stream_response("disconnect", "u1", "resume-run", []) if resume
              else module.get_agent_stream_response("question", "disconnect", "u1"))
    pending = None
    try:
        await anext(stream)
        await asyncio.wait_for(started.wait(), 1)
        if closing:
            await stream.aclose()
        else:
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert cancelled.is_set()
        assert all(task.done() for task in tasks)
        assert not module._get_session_lock("u1", "disconnect").locked()
        assert len(deleted) == (0 if resume else 1)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if pending:
            await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()


@pytest.mark.parametrize("resume", [False, True])
async def test_run_timeout_cancels_work_emits_error_then_done_and_releases_lock(monkeypatch, resume):
    cancelled = asyncio.Event()
    deleted, saved = [], []

    class Manager:
        async def get_pending_run_id(self, *args):
            return "resume-run" if resume else None

        async def get_history(self, *args):
            return []

        async def add_message(self, *args, **kwargs):
            saved.append(args)

    async def emit(agent, inputs, config, queue, full_response):
        full_response.append("未完成的回答")
        await queue.put({"type": "response", "content": "未完成的回答"})
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def delete(run_id):
        deleted.append(run_id)

    monkeypatch.setattr(module.settings, "AGENT_RUN_TIMEOUT_SECONDS", .02)
    monkeypatch.setattr(module, "sm", SimpleNamespace(session_manager=Manager()))
    monkeypatch.setattr(module.agent_factory, "create_agent", lambda **kwargs: object())
    monkeypatch.setattr(module, "_emit_agent_events", emit)
    monkeypatch.setattr(module, "get_checkpointer", lambda: SimpleNamespace(adelete_thread=delete))
    stream = (module.get_agent_resume_stream_response("timeout", "u1", "resume-run", []) if resume
              else module.get_agent_stream_response("question", "timeout", "u1"))
    events = [json.loads(frame.removeprefix("data: ")) async for frame in stream]
    assert events[-2]["type"] == "error" and "超时" in events[-2]["content"]
    assert events[-1]["type"] == "done" and cancelled.is_set()
    assert any(event.get("content") == "未完成的回答" for event in events)
    assert not saved and not module._get_session_lock("u1", "timeout").locked()
    assert len(deleted) == (0 if resume else 1)
