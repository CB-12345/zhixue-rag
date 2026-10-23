"""基础索引与图谱独立消费的边界测试（隔离 SQLite + 真实临时 Neo4j）。"""
import asyncio
import os

import pytest
from sqlalchemy import select

from app.core.background_init import init_manager
from app.graph.schemas.graph import ExtractResult
from app.graph.services import graph_service
from app.graph.services.graph_worker import _claim_next_task, _tick, recover_stale_tasks
from app.graph.storage import neo4j_client
from app.models.graph import GraphBuildTask
from tests.fakes import make_fake_chat_model
from tests.graph.test_neo4j_pipeline import FakeEmbed


@pytest.fixture
def pipeline(session_factory, monkeypatch):
    monkeypatch.setattr(graph_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(init_manager, "embed_model", FakeEmbed())
    monkeypatch.setattr(init_manager, "chat_model", None)
    return session_factory


@pytest.fixture
def neo4j_env(monkeypatch):
    if not os.getenv("NEO4J_TEST_URI"):
        pytest.skip("需要独立 Neo4j 测试实例")
    from app.core.settings import settings

    monkeypatch.setattr(settings, "NEO4J_URI", os.environ["NEO4J_TEST_URI"])


async def chunks_for(doc_id):
    rows = await neo4j_client.get_neo4j_driver().execute_query(
        "MATCH (c:Chunk {user_id:'pipe-index-user', source_id:$id}) "
        "RETURN c.text AS text, c.embedding AS embedding ORDER BY c.chunk_index", {"id": doc_id})
    return [dict(r) for r in rows.records]


async def task_for(factory, doc_id):
    async with factory() as db:
        return (await db.execute(select(GraphBuildTask).where(
            GraphBuildTask.source_id == doc_id))).scalar_one()


async def enqueue(doc_id, text="基础检索事实"):
    await graph_service.maybe_schedule_doc_extraction(
        "pipe-index-user", doc_id, doc_id + ".pdf", text,
        chunks=[{"chunk_index": 0, "text": text, "page": 2, "image_paths": []}])


async def test_upload_conflict_delete_reupload_and_cross_user_index_isolation(
        pipeline, monkeypatch, neo4j_env, _cleanup):
    from fastapi import HTTPException

    from app.rag.md5_manager import MD5Store
    from app.rag.vector_store import VectorStoreService

    one, two = "pipe-upload-one", "pipe-upload-two"
    old = [{"chunk_index": 0, "text": "旧编号 OLD-UPLOAD-2048", "page": 2}]
    new = [{"chunk_index": 0, "text": "新编号 NEW-UPLOAD-4096", "page": 3}]
    for owner in (one, two):
        assert await graph_service.register_document_upload(owner, "same-bytes", "更新.txt", old[0]["text"], old)
        assert await _tick("index")
    driver = neo4j_client.get_neo4j_driver()

    async def texts(owner):
        result = await driver.execute_query(
            "MATCH (c:Chunk {user_id:$uid}) RETURN c.text AS text", {"uid": owner})
        return [row["text"] for row in result.records]

    with pytest.raises(HTTPException) as exc:
        await graph_service.register_document_upload(one, "new-bytes", "更新.txt", new[0]["text"], new)
    assert exc.value.status_code == 409
    assert await texts(one) == [old[0]["text"]]
    # 不依赖 MD5 文件也可以删除；另一用户相同内容的 Doc/Chunk 保持完整。
    monkeypatch.setattr("app.db.db_config.AsyncSessionLocal", pipeline)
    service = object.__new__(VectorStoreService)
    service.md5_store = MD5Store()
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        service.md5_store.base_dir = root
        assert await service.delete_by_filename(one, "更新.txt")
        assert await texts(one) == []
        assert await texts(two) == [old[0]["text"]]
        assert await graph_service.register_document_upload(one, "new-bytes", "更新.txt", new[0]["text"], new)
        assert await _tick("index")
        assert await texts(one) == [new[0]["text"]]
        assert await texts(two) == [old[0]["text"]]
        detail = await service.get_document_detail(one, "更新.txt")
        assert detail["chunks"][0]["page"] == 3 and detail["chunk_count"] == 1


async def test_index_works_without_chat_graph_failure_and_retry_keep_vectors(
        pipeline, monkeypatch, neo4j_env, _cleanup):
    class CountingEmbed(FakeEmbed):
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return super().embed_documents(texts)

    embed = CountingEmbed()
    monkeypatch.setattr(init_manager, "embed_model", embed)
    await enqueue("independent")
    assert await _claim_next_task() is None  # 图谱不能越过基础索引
    assert await _tick("index")
    before = await chunks_for("independent")
    assert before[0]["text"] == "基础检索事实" and len(before[0]["embedding"]) == 8
    task = await task_for(pipeline, "independent")
    assert task.index_status == "completed" and task.status == "pending"

    for _ in range(graph_service.MAX_TASK_ATTEMPTS):
        assert await _tick("graph")  # 缺少聊天模型只影响图谱
    task = await task_for(pipeline, "independent")
    assert task.status == "failed" and task.index_status == "completed"
    assert await chunks_for("independent") == before
    from app.rag.vector_store import VectorStoreService
    from app.schemas.models import KnowledgeListResponse

    monkeypatch.setattr("app.db.db_config.AsyncSessionLocal", pipeline)
    service = object.__new__(VectorStoreService)
    docs = await service.get_user_documents("pipe-index-user")
    response = KnowledgeListResponse(documents=docs, total_count=len(docs))
    assert response.documents[0].status == "completed"
    assert response.documents[0].graph_status == "failed"
    assert response.documents[0].error_message is None
    await graph_service.retry_document("pipe-index-user", "independent")
    assert not await _tick("index")  # 图谱重试不再次计算向量
    monkeypatch.setattr(init_manager, "chat_model", make_fake_chat_model([
        '{"entities":[],"relations":[]}']))
    assert await _tick("graph")
    assert (await task_for(pipeline, "independent")).status == "completed"
    assert embed.calls == 1 and await chunks_for("independent") == before


async def test_slow_graph_does_not_block_next_document_index(
        pipeline, monkeypatch, neo4j_env, _cleanup):
    started, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def slow_extract(title, body, model):
        seen.append(body)
        started.set()
        await release.wait()
        return ExtractResult(entities=[], relations=[])

    monkeypatch.setattr(init_manager, "chat_model", object())
    monkeypatch.setattr(graph_service, "extract_entities", slow_extract)
    await enqueue("slow-graph", "第一份正文")
    assert await _tick("index")
    graph_run = asyncio.create_task(_tick("graph"))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        await enqueue("next-doc", "第二份可检索正文")
        assert await asyncio.wait_for(_tick("index"), timeout=5)
        assert (await chunks_for("next-doc"))[0]["text"] == "第二份可检索正文"
        assert (await task_for(pipeline, "slow-graph")).status == "running"
        assert seen == ["第一份正文"]  # 预切文档仍把完整正文传给实体抽取
    finally:
        release.set()
        await asyncio.wait_for(graph_run, timeout=5)


async def test_index_failure_preserves_previous_chunks_and_retry_releases_graph(
        pipeline, monkeypatch, neo4j_env, _cleanup):
    await enqueue("index-retry", "原有正文")
    assert await _tick("index")
    before = await chunks_for("index-retry")
    monkeypatch.setattr(init_manager, "embed_model", None)
    await enqueue("index-retry", "新正文")
    for _ in range(graph_service.MAX_TASK_ATTEMPTS):
        assert await _tick("index")
    assert (await task_for(pipeline, "index-retry")).index_status == "failed"
    assert await chunks_for("index-retry") == before
    assert await _claim_next_task("graph") is None
    await graph_service.retry_document("pipe-index-user", "index-retry")
    monkeypatch.setattr(init_manager, "embed_model", FakeEmbed())
    assert await _tick("index")
    assert (await chunks_for("index-retry"))[0]["text"] == "新正文"
    assert await _claim_next_task("graph") is not None


async def test_delete_during_embedding_does_not_restore_document(
        pipeline, monkeypatch, neo4j_env, _cleanup):
    loop = asyncio.get_running_loop()

    class DeletingEmbed(FakeEmbed):
        def embed_documents(self, texts):
            asyncio.run_coroutine_threadsafe(
                graph_service.cleanup_doc_graph("pipe-index-user", "deleted"), loop).result(timeout=5)
            return super().embed_documents(texts)

    await enqueue("deleted")
    monkeypatch.setattr(init_manager, "embed_model", DeletingEmbed())
    assert await _tick("index")
    assert await chunks_for("deleted") == []
    async with pipeline() as db:
        assert (await db.execute(select(GraphBuildTask))).scalars().all() == []
    assert await _claim_next_task("graph") is None


async def test_both_running_stages_recover_independently(pipeline):
    await enqueue("recover")
    task_id, token = await _claim_next_task("index")
    assert token
    assert await recover_stale_tasks() == 1
    task = await task_for(pipeline, "recover")
    assert task.id == task_id and task.index_status == "pending" and task.index_run_token is None
    async with pipeline() as db:
        task = await db.get(GraphBuildTask, task_id)
        task.index_status, task.indexed_hash = "completed", task.content_hash
        await db.commit()
    assert await _claim_next_task("graph") is not None
    assert await recover_stale_tasks() == 1
    task = await task_for(pipeline, "recover")
    assert task.status == "pending" and task.index_status == "completed" and task.run_token is None
    assert await recover_stale_tasks() == 0


async def test_legacy_chunks_only_payload_supplies_graph_body(
        pipeline, monkeypatch, neo4j_env, _cleanup):
    import json

    seen = []

    async def extract(title, body, model):
        seen.append(body)
        return ExtractResult(entities=[], relations=[])

    await enqueue("legacy-payload", "旧版本预切正文")
    async with pipeline() as db:
        task = (await db.execute(select(GraphBuildTask))).scalar_one()
        payload = json.loads(task.payload)
        del payload["text"]
        task.payload = json.dumps(payload)
        await db.commit()
    monkeypatch.setattr(init_manager, "chat_model", object())
    monkeypatch.setattr(graph_service, "extract_entities", extract)
    assert await _tick("index")
    assert await _tick("graph")
    assert seen == ["旧版本预切正文"]


async def test_edit_during_graph_discards_old_entities(
        pipeline, monkeypatch, neo4j_env, _cleanup):
    from app.graph.schemas.graph import ExtractedEntity

    async def editing_extract(title, body, model):
        await enqueue("graph-edit", "最新正文")
        return ExtractResult(entities=[ExtractedEntity(name="过时实体", mentions=["过时实体"])], relations=[])

    monkeypatch.setattr(init_manager, "chat_model", object())
    monkeypatch.setattr(graph_service, "extract_entities", editing_extract)
    await enqueue("graph-edit", "过时实体")
    assert await _tick("index")
    assert await _tick("graph")
    task = await task_for(pipeline, "graph-edit")
    assert task.status == "pending" and task.index_status == "pending"
    result = await neo4j_client.get_neo4j_driver().execute_query(
        "MATCH (e:Entity {user_id:'pipe-index-user',name:'过时实体'}) RETURN count(e) AS count")
    assert result.records[0]["count"] == 0
    assert await _tick("index")
    assert (await chunks_for("graph-edit"))[0]["text"] == "最新正文"
