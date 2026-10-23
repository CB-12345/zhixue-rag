"""已有任务升级时保留完成的索引，并恢复未完成的基础索引。"""
from sqlalchemy import Column, MetaData, Table, select

from app.db.db_config import _migrate_columns
from app.models.graph import GraphBuildTask


async def test_legacy_task_migration_preserves_completed_and_is_idempotent(db_engine):
    metadata = MetaData()
    legacy = Table("graph_build_tasks", metadata, *[
        Column(c.name, c.type, primary_key=c.primary_key, nullable=c.nullable)
        for c in GraphBuildTask.__table__.columns
        if not c.name.startswith("index_") and c.name != "indexed_hash"])
    async with db_engine.begin() as conn:
        await conn.run_sync(GraphBuildTask.__table__.drop)
        await conn.run_sync(metadata.create_all)
        await conn.execute(legacy.insert(), [
            {"id": status, "user_id": "legacy-user", "source_type": "doc", "source_id": status,
             "title": "旧资料", "content_hash": "hash-" + status, "payload": '{"chunks":[]}',
             "status": status, "attempts": 1, "force": False}
            for status in ("completed", "pending", "running", "failed")])
        await _migrate_columns(conn)
        before = (await conn.execute(select(GraphBuildTask.__table__))).mappings().all()
        await _migrate_columns(conn)
        after = (await conn.execute(select(GraphBuildTask.__table__))).mappings().all()
    assert before == after
    for task in after:
        assert task["payload"] == '{"chunks":[]}'
        assert task["index_attempts"] == 0
        if task["status"] == "completed":
            assert task["index_status"] == "completed" and task["indexed_hash"] == "hash-completed"
        else:
            assert task["index_status"] == "pending" and task["indexed_hash"] is None
