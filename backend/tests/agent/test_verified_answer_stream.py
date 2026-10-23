import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage

from app.agent import agent as mod
from app.agent.agent_rag_tool import _register_supplementary_sources
from app.rag.agentic_rag.schemas import Evidence
from app.rag.answer_verifier import VerifiedAnswer
from tests.fakes import FakeAgent

SOURCE = {"number": 1, "id": "doc:fixture:0", "source": "knowledge_base", "title": "课表", "content": "周四开放。"}


def setup(monkeypatch, draft, verifier, events=None):
    manager = SimpleNamespace(get_pending_run_id=AsyncMock(return_value=None), get_history=AsyncMock(return_value=[]), add_message=AsyncMock())
    fake = FakeAgent(messages=[AIMessage(content=draft)], events=events or [
        {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content=draft)}}])
    monkeypatch.setattr(mod, "sm", SimpleNamespace(session_manager=manager))
    monkeypatch.setattr(mod.agent_factory, "create_agent", lambda **kw: fake)
    monkeypatch.setattr(mod, "get_checkpointer", lambda: SimpleNamespace(adelete_thread=AsyncMock()))
    monkeypatch.setattr(mod, "AnswerVerifier", lambda: SimpleNamespace(verify=verifier))
    return manager


async def consume(events, **kwargs):
    async for frame in mod.get_agent_stream_response("开放时间？", "verify-session", "verify-user", rag_context="[1] 周四开放。", rag_sources=[SOURCE], **kwargs):
        events.append(json.loads(frame.removeprefix("data: ")))


async def test_draft_is_neither_published_nor_saved_until_review_finishes(monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def verify(query, draft, sources):
        assert draft == "周四开放。不需要登记[1]。" and sources == [SOURCE]
        started.set()
        await release.wait()
        return VerifiedAnswer("周四开放[1]。", "filtered", 1, 1)

    manager = setup(monkeypatch, "周四开放。不需要登记[1]。", verify)
    events = []
    task = asyncio.create_task(consume(events))
    await started.wait()
    assert all(not e.get("content") for e in events if e["type"] == "response")
    manager.add_message.assert_not_called()
    release.set()
    await task
    visible = "".join(e.get("content", "") for e in events if e["type"] == "response")
    assert visible == "周四开放[1]。" and events[-1]["type"] == "done"
    assert manager.add_message.call_args.args[-1] == visible
    assert manager.add_message.call_args.kwargs["sources"] == [SOURCE]


async def test_final_snapshot_overrides_nested_model_tokens(monkeypatch):
    verifier = AsyncMock(return_value=VerifiedAnswer("周四开放[1]。", "supported", 1))
    setup(monkeypatch, "周四开放[1]。", verifier, events=[
        {"event": "on_chat_model_stream", "data": {"chunk": AIMessage(content="内部规划和未核对内容")}}])
    events = []
    await consume(events)
    assert verifier.call_args.args[1] == "周四开放[1]。"
    assert "内部规划" not in str(events)


async def test_disconnect_during_verification_cancels_work_without_saving_draft(monkeypatch):
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def verify(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    manager = setup(monkeypatch, "草稿", verify)
    events = []
    task = asyncio.create_task(consume(events))
    await started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert cancelled.is_set() and not mod._get_session_lock("verify-user", "verify-session").locked()
    manager.add_message.assert_not_called()


def test_supplementary_evidence_uses_one_stable_numbering_scheme():
    registry = [dict(SOURCE)]
    context = _register_supplementary_sources([
        Evidence(id=SOURCE["id"], source="knowledge_base", title="课表", content=SOURCE["content"]),
        Evidence(id="doc:fixture:1", source="knowledge_base", title="课表", content="周五开放。", metadata={"page": 3}),
    ], registry)
    assert "[1] 来源" in context and "[2] 来源" in context and "第 3 页" in context
    assert [s["number"] for s in registry] == [1, 2]
    again = _register_supplementary_sources([Evidence(id="doc:fixture:1", source="knowledge_base", title="课表", content="周五开放。", metadata={"page": 3})], registry)
    assert again.startswith("[2]") and len(registry) == 2


async def test_operation_receipts_preserve_existing_flow_without_claiming_document_verification(monkeypatch):
    verifier = AsyncMock()
    manager = setup(monkeypatch, "已创建笔记。", verifier, events=[{"event": "on_tool_start", "name": "create_note_tool", "data": {}}])
    events = []
    await consume(events)
    verifier.assert_not_called()
    assert manager.add_message.call_args.args[-1] == "已创建笔记。"
    assert not any(e.get("stage") == "answer_verification" for e in events)
