"""真实 MySQL 的主键迁移与并发上传；仅连接显式指定的独立测试数据库。"""
import asyncio
import os

import pytest
from fastapi import HTTPException
from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.db_config import _migrate_columns
from app.graph.services import graph_service
from app.models.chat_history import Base
from app.models.graph import GraphBuildTask, GraphDoc
from app.models.user_model import User

pytestmark = pytest.mark.skipif(not os.getenv("MYSQL_TEST_URL"), reason="需要独立 MYSQL_TEST_URL")


@pytest.fixture
async def mysql_factory(monkeypatch):
    engine = create_async_engine(os.environ["MYSQL_TEST_URL"])
    # 独立容器里的独立数据库；防止误指向应用数据库。
    assert engine.url.database == "upload_tests"
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("DELETE FROM graph_build_tasks"))
        await conn.execute(text("DELETE FROM graph_docs"))
        await conn.execute(User.__table__.delete())
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(graph_service, "AsyncSessionLocal", factory)
    yield engine, factory
    await engine.dispose()


async def test_legacy_primary_key_migration_preserves_rows_and_is_idempotent(mysql_factory):
    engine, factory = mysql_factory
    async with engine.begin() as conn:
        await conn.run_sync(GraphDoc.__table__.drop)
        await conn.execute(text(
            "CREATE TABLE graph_docs (id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) NOT NULL, "
            "filename VARCHAR(255) NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
        await conn.execute(text(
            "INSERT INTO graph_docs(id,user_id,filename) VALUES ('same-md5','first-owner','old.txt')"))
        before = (await conn.execute(text("SELECT * FROM graph_docs"))).mappings().all()
        await _migrate_columns(conn)
        await _migrate_columns(conn)
        after = (await conn.execute(text("SELECT * FROM graph_docs"))).mappings().all()
        keys = await conn.run_sync(lambda c: inspect(c).get_pk_constraint("graph_docs")["constrained_columns"])
    assert before == after and keys == ["id", "user_id"]
    async with factory() as db:
        db.add(GraphDoc(id="same-md5", user_id="second-owner", filename="copy.txt"))
        await db.commit()
        assert len((await db.execute(select(GraphDoc))).scalars().all()) == 2


@pytest.mark.parametrize("same_content", [False, True])
async def test_concurrent_uploads_are_serialized_by_user(mysql_factory, same_content):
    _, factory = mysql_factory
    async with factory() as db:
        db.add(User(uuid="concurrent-owner", username="upload-test", email="upload@example.test",
                    password="test-placeholder"))
        await db.commit()
    second_id = "one-md5" if same_content else "two-md5"
    results = await asyncio.gather(*[
        graph_service.register_document_upload("concurrent-owner", doc_id, filename, "正文", [])
        for doc_id, filename in [("one-md5", "RAG.txt"), (second_id, "RAG.txt" if same_content else "rag.txt")]
    ], return_exceptions=True)
    assert sum(r is True for r in results) == 1
    if same_content:
        assert sum(r is False for r in results) == 1
    else:
        conflicts = [r for r in results if isinstance(r, HTTPException)]
        assert len(conflicts) == 1 and conflicts[0].status_code == 409
    async with factory() as db:
        assert len((await db.execute(select(GraphDoc))).scalars().all()) == 1
        assert len((await db.execute(select(GraphBuildTask))).scalars().all()) == 1
