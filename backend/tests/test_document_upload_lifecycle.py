"""上传身份、事务与删除的回归测试；SQLite + 真实 TXT 解析，不调用外部模型。"""
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import select

from app.graph.services import graph_service
from app.graph.storage.neo4j_client import GraphUnavailableError
from app.models.graph import GraphBuildTask, GraphDoc
from app.rag.document_handler.processor import DocumentProcessor
from app.rag.md5_manager import MD5Store
from app.rag.vector_store import VectorStoreService
from app.router.knowledge_service import KnowledgeService


@pytest.fixture
def upload_env(session_factory, monkeypatch, tmp_path):
    monkeypatch.setattr(graph_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr("app.db.db_config.AsyncSessionLocal", session_factory)
    store = object.__new__(VectorStoreService)
    store.md5_store = MD5Store()
    store.md5_store.base_dir = str(tmp_path / "records")
    store.document_processor = DocumentProcessor(store.md5_store)
    monkeypatch.setattr("app.router.knowledge_service.VectorStoreService", lambda: store)
    monkeypatch.setattr("app.rag.vector_store.delete_image_directory", lambda *_: None)
    return KnowledgeService(), store, session_factory


def file(name="资料.txt", text="UPLOAD-2048 的负责人是林知夏。"):
    return UploadFile(filename=name, file=BytesIO(text.encode("utf-8")))


async def upload(service, files=None, user="owner"):
    events = [json.loads(event.split("data: ", 1)[1]) async for event in
              service.handle_add_vector_multiple_stream(files or [file()], user)]
    return events


async def rows(factory, model):
    async with factory() as db:
        return list((await db.execute(select(model))).scalars())


@pytest.mark.parametrize("status", ["pending", "running", "completed", "failed"])
async def test_repeat_upload_preserves_task_progress_and_does_not_append_records(upload_env, status):
    service, store, factory = upload_env
    first = await upload(service)
    assert first[-2]["step"] == "queued"
    async with factory() as db:
        task = (await db.execute(select(GraphBuildTask))).scalar_one()
        task.index_status = task.status = status
        task.index_attempts = task.attempts = 2
        task.index_run_token = task.run_token = "keep-this-claim"
        await db.commit()
    again = await upload(service)
    assert again[-2]["step"] == "existing"
    task, = await rows(factory, GraphBuildTask)
    assert task.index_status == task.status == status
    assert task.index_attempts == task.attempts == 2
    assert task.index_run_token == task.run_token == "keep-this-claim"
    assert len(await rows(factory, GraphDoc)) == 1
    assert len(await store.md5_store.get_all_md5_records("owner")) == 1


async def test_same_name_changed_content_returns_conflict_in_both_interfaces(upload_env):
    service, store, factory = upload_env
    await upload(service)
    original, = await rows(factory, GraphBuildTask)
    events = await upload(service, [file(text="全新内容，负责人变更。")])
    error = next(e for e in events if e["event_type"] == "error")
    assert error["status_code"] == 409 and "先删除" in error["error_message"]
    assert events[-1]["failed_count"] == 1 and events[-1]["success_count"] == 0
    with pytest.raises(HTTPException) as exc:
        await service.handle_add_vector_single(file(text="另一份内容"), "owner")
    assert exc.value.status_code == 409
    task, = await rows(factory, GraphBuildTask)
    assert task.id == original.id and task.payload == original.payload
    assert len(await rows(factory, GraphDoc)) == 1
    assert len(await store.md5_store.get_all_md5_records("owner")) == 1


async def test_same_content_different_name_preserves_original_source_name(upload_env):
    service, _, factory = upload_env
    await upload(service)
    events = await upload(service, [file("改名.txt")])
    assert events[-2]["status_code"] == 409
    doc, = await rows(factory, GraphDoc)
    assert doc.filename == "资料.txt"


async def test_identical_bytes_are_independent_per_user(upload_env):
    service, store, factory = upload_env
    await upload(service)
    await upload(service, user="other-owner")
    docs = await rows(factory, GraphDoc)
    assert len(docs) == 2 and docs[0].id == docs[1].id
    assert {d.user_id for d in docs} == {"owner", "other-owner"}
    assert len(await rows(factory, GraphBuildTask)) == 2
    assert len(await store.get_all_md5_records("other-owner")) == 1


async def test_failed_task_registration_rolls_back_and_reupload_recovers(upload_env, monkeypatch):
    service, store, factory = upload_env
    original = graph_service._upsert_build_task
    monkeypatch.setattr(graph_service, "_upsert_build_task", AsyncMock(side_effect=RuntimeError("db failure")))
    events = await upload(service)
    assert events[-2]["status_code"] == 503
    assert await rows(factory, GraphDoc) == []
    assert await rows(factory, GraphBuildTask) == []
    assert await store.md5_store.get_all_md5_records("owner") == []
    monkeypatch.setattr(graph_service, "_upsert_build_task", original)
    assert (await upload(service))[-2]["step"] == "queued"
    assert len(await rows(factory, GraphBuildTask)) == 1


async def test_legacy_md5_marker_cannot_block_missing_task_registration(upload_env):
    import hashlib
    service, store, factory = upload_env
    content = "UPLOAD-2048 的负责人是林知夏。".encode()
    await store.md5_store.save_md5_hex(hashlib.md5(content).hexdigest(), "资料.txt", "资料.txt", "owner")
    assert (await upload(service))[-2]["step"] == "queued"
    assert len(await rows(factory, GraphDoc)) == len(await rows(factory, GraphBuildTask)) == 1
    assert len(await store.md5_store.get_all_md5_records("owner")) == 1


async def test_registry_without_task_is_repaired(upload_env):
    import hashlib
    service, _, factory = upload_env
    md5 = hashlib.md5("UPLOAD-2048 的负责人是林知夏。".encode()).hexdigest()
    async with factory() as db:
        db.add(GraphDoc(id=md5, user_id="owner", filename="资料.txt"))
        await db.commit()
    assert (await upload(service))[-2]["step"] == "queued"
    assert len(await rows(factory, GraphDoc)) == len(await rows(factory, GraphBuildTask)) == 1


async def test_compatibility_write_failure_keeps_durable_upload_and_metadata(upload_env, monkeypatch):
    service, store, factory = upload_env
    monkeypatch.setattr(store, "save_md5_hex", AsyncMock(side_effect=OSError("disk full")))
    events = await upload(service)
    assert events[-1]["success_count"] == 1
    assert await store.md5_store.get_all_md5_records("owner") == []
    doc, = await rows(factory, GraphDoc)
    assert (await store.get_md5_info("owner", doc.id))["filename"] == "资料.txt"
    assert await store.get_all_md5_records("stranger") == []


async def test_batch_validation_failure_is_included_in_final_totals(upload_env):
    service, _, _ = upload_env
    events = await upload(service, [file("empty.txt", ""), file()])
    assert events[-1]["failed_count"] == events[-1]["success_count"] == 1


async def test_delete_storage_failure_keeps_registration_and_records_for_retry(upload_env, monkeypatch):
    service, store, factory = upload_env
    await upload(service)
    graph = SimpleNamespace(clear_source_data=AsyncMock(side_effect=GraphUnavailableError("offline")),
                            sweep_orphan_entities=AsyncMock())
    monkeypatch.setattr(graph_service, "get_graph_store", lambda db: graph)
    with pytest.raises(HTTPException) as exc:
        await store.delete_by_filename("owner", "资料.txt")
    assert exc.value.status_code == 503
    assert len(await rows(factory, GraphDoc)) == 1
    assert len(await store.md5_store.get_all_md5_records("owner")) == 1
    graph.clear_source_data = AsyncMock(return_value=[])
    assert await store.delete_by_filename("owner", "资料.txt")
    assert await rows(factory, GraphDoc) == []
    assert await store.md5_store.get_all_md5_records("owner") == []


async def test_delete_all_legacy_same_name_revisions_and_preserve_other_user(upload_env, monkeypatch):
    _, store, factory = upload_env
    async with factory() as db:
        db.add_all([GraphDoc(id=source, user_id=owner, filename="历史.txt")
                    for source, owner in [("v1", "owner"), ("v2", "owner"), ("v1", "stranger")]])
        await db.commit()
    graph = SimpleNamespace(clear_source_data=AsyncMock(return_value=[]), sweep_orphan_entities=AsyncMock())
    monkeypatch.setattr(graph_service, "get_graph_store", lambda db: graph)
    assert await store.delete_by_filename("owner", "历史.txt")
    assert [c.args[2] for c in graph.clear_source_data.await_args_list] == ["v1", "v2"]
    doc, = await rows(factory, GraphDoc)
    assert doc.user_id == "stranger" and doc.id == "v1"
