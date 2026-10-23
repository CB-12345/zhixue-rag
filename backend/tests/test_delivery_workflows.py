"""资料溯源与任务重试：真实数据库边界测试，不调用外部模型。"""
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.models.graph import GraphBuildTask
from app.rag.agentic_rag.schemas import Evidence
from app.rag.agentic_rag.sources import public_sources
from app.services.database_session_manager import DatabaseSessionManager
from tests.conftest import patch_session_factory


async def test_answer_sources_survive_reload_and_are_owner_scoped(session_factory, monkeypatch):
    patch_session_factory(monkeypatch, session_factory)
    manager = DatabaseSessionManager()
    sources = public_sources([Evidence(id="c1", source="note", title="学习记录", content="混合检索",
                                      metadata={"source_id": "n1", "chunk_index": 0, "page": 2, "secret": "hidden"})])
    await manager.add_message("sources-session", "owner", "问一", "答一", sources=sources)
    await manager.add_message("sources-session", "owner", "问二", "答二")
    restored = await manager.get_session("sources-session", "owner", include_sources=True)
    assert restored["history"] == [("问一", "答一"), ("问二", "答二")]
    assert restored["sources"] == [sources, []]
    assert restored["sources"][0][0]["page"] == 2
    assert "secret" not in json.dumps(restored)
    with pytest.raises(HTTPException) as error:
        await manager.get_session("sources-session", "stranger", include_sources=True)
    assert error.value.status_code == 403


def test_sources_reject_unsafe_links_and_bound_snapshots():
    result = public_sources([Evidence(id="w", source="web", title="页面", content="x" * 3000,
                                     url="javascript:alert(1)")])
    assert result[0]["url"] is None
    assert len(result[0]["content"]) == 1600


async def test_sources_survive_approval_resume(session_factory, monkeypatch):
    patch_session_factory(monkeypatch, session_factory)
    manager = DatabaseSessionManager()
    sources = [{"title": "资料", "source": "note", "content": "原文"}]
    await manager.get_session("approval-session", "owner")
    await manager.set_pending_run_id("approval-session", "owner", "run1", sources=sources)
    await manager.add_message("approval-session", "owner", "整理成笔记", "已创建笔记")
    restored = await manager.get_session("approval-session", "owner", include_sources=True)
    assert restored["sources"] == [sources]


async def test_retry_requires_failed_state_and_owner_preserving_payload(session_factory, monkeypatch):
    from app.graph.services import graph_service

    monkeypatch.setattr(graph_service, "AsyncSessionLocal", session_factory)
    async with session_factory() as db:
        db.add(GraphBuildTask(id="retry-task", user_id="owner", source_type="doc", source_id="doc1",
                              title="资料.md", content_hash="hash", payload='{"chunks": [{"text": "原文"}]}',
                              status="failed", attempts=3, error_message="failed"))
        await db.commit()
    with pytest.raises(HTTPException):
        await graph_service.retry_document("stranger", "doc1")
    await graph_service.retry_document("owner", "doc1")
    async with session_factory() as db:
        task = (await db.execute(select(GraphBuildTask))).scalar_one()
        assert task.status == "pending"
        assert task.attempts == 0
        assert task.force is True
        assert json.loads(task.payload)["chunks"][0]["text"] == "原文"
    with pytest.raises(HTTPException):
        await graph_service.retry_document("owner", "doc1")
