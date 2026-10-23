"""知识库文档服务（文档列表/详情/切片/MD5/上传管线）。

- chunk/向量统一存 Neo4j（(:Chunk) 节点，由图谱构建 worker 写入，见 graph_service）
- 本服务保留：文档列表/详情/切片查询（读 Neo4j）、md5 记录（上传去重）、文档删除联动
- 文档加载与切片（DocumentProcessor）保留：上传管线用它产出任务载荷
"""
import asyncio
import threading

from langchain_core.documents import Document

from app.core.logger_handler import logger
from app.utils.image_extractor import delete_image_directory, delete_user_all_images

from .document_handler import DocumentProcessor
from .md5_manager import MD5Store


class VectorStoreService:
    """知识库文档服务（单例，线程安全初始化）。"""
    _instance = None
    _initialized = False
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if VectorStoreService._initialized:
            return

        with VectorStoreService._init_lock:
            if VectorStoreService._initialized:
                return

            self.md5_store = MD5Store()
            self.document_processor = DocumentProcessor(self.md5_store)
            VectorStoreService._initialized = True

    # ---- 文档列表/详情/切片（读 Neo4j） ----
    @staticmethod
    def _neo4j_driver():
        # 独立访问点：测试可替换
        from app.graph.storage.neo4j_client import get_neo4j_driver

        return get_neo4j_driver()

    @classmethod
    async def _query_neo4j(cls, query: str, params: dict | None = None):
        """执行 Neo4j 查询并取回 records（本服务读文档数据的统一入口）。"""
        driver = cls._neo4j_driver()
        result = await driver.execute_query(query, params or {})
        return list(result.records)

    @staticmethod
    def _chunk_image_urls(md5: str, image_paths) -> list[str]:
        """chunk 图片相对路径 → 前端可访问的 /knowledge/image URL 列表。"""
        if isinstance(image_paths, list) and image_paths:
            return [f"/knowledge/image/{md5}/{img}" for img in image_paths]
        return []

    async def get_user_documents(self, user_id: str = None):
        """获取用户的知识库文档列表（MySQL graph_docs 注册行 ∪ Neo4j Doc/Chunk 聚合）。

        graph_docs 在上传入队时即登记，是文档存在性的即时基准——新上传的文档无需等待
        图谱抽取完成即可出现在列表中；chunk 数与预览来自 Neo4j，抽取未完成时为 0。
        created_at 优先取注册行的真实上传时间。
        """
        # 1. MySQL 注册行（上传即写入）
        registered: dict[str, dict] = {}
        try:
            from sqlalchemy import select

            from app.db.db_config import AsyncSessionLocal
            from app.models.graph import GraphDoc

            async with AsyncSessionLocal() as db:
                stmt = select(GraphDoc.id, GraphDoc.filename, GraphDoc.created_at)
                if user_id is not None:
                    stmt = stmt.where(GraphDoc.user_id == user_id)
                rows = (await db.execute(stmt.order_by(GraphDoc.created_at))).all()
            for rid, filename, created_at in rows:
                registered[rid] = {"filename": filename, "created_at": created_at}
        except Exception as e:
            logger.error(f"【知识库】读取文档注册行失败: {e}")

        # 2. Neo4j Doc/Chunk 聚合（抽取完成后才有；失败不阻塞注册行展示）
        try:
            records = await self._query_neo4j(
                "MATCH (d:Doc) WHERE $uid IS NULL OR d.user_id = $uid "
                "OPTIONAL MATCH (c:Chunk {kind: 'doc'}) "
                "WHERE c.source_id = d.id AND c.user_id = d.user_id "
                "WITH d, count(c) AS chunk_count, head(collect(c.text)) AS first_text "
                "RETURN d.id AS id, d.filename AS filename, d.user_id AS user_id, "
                "chunk_count, first_text",
                {"uid": user_id},
            )
        except Exception as e:
            logger.error(f"【知识库】获取用户文档列表失败: {e}")
            records = []

        preview_length = 100
        task_states = {}
        try:
            from app.models.graph import GraphBuildTask

            async with AsyncSessionLocal() as db:
                stmt = select(GraphBuildTask).where(
                    GraphBuildTask.source_type == "doc")
                if user_id is not None:
                    stmt = stmt.where(GraphBuildTask.user_id == user_id)
                for task in (await db.execute(stmt)).scalars():
                    task_states[task.source_id] = {
                        "status": task.index_status, "attempts": task.index_attempts,
                        "graph_status": task.status if task.index_status == "completed" else "waiting",
                        "graph_attempts": task.attempts,
                        "error_message": "索引失败，请检查向量模型连接后重试" if task.index_status == "failed" else None,
                        "graph_error_message": "图谱构建失败，基础检索仍可使用" if task.status == "failed" else None,
                    }
        except Exception as e:
            logger.error(f"【知识库】读取处理状态失败: {e}")

        def _entry(doc_id: str, filename: str, user, chunk_count: int, preview: str, created_at) -> dict:
            return {
                "id": doc_id,
                "filename": filename,
                "original_filename": filename,
                "user_id": user,
                "chunk_count": chunk_count,
                "preview": (preview or "")[:preview_length] + ("..." if len(preview or "") > preview_length else ""),
                "created_at": created_at.isoformat() if created_at else None,
                **task_states.get(doc_id, {"status": "completed" if chunk_count else "unknown",
                                          "attempts": 0, "graph_status": "unknown"}),
            }

        docs_info: list[dict] = []
        seen_ids: set[str] = set()
        for row in records:
            doc_id = row.get("id")
            seen_ids.add(doc_id)
            reg = registered.get(doc_id, {})
            docs_info.append(_entry(
                doc_id, reg.get("filename") or row.get("filename") or "unknown",
                row.get("user_id"), row.get("chunk_count") or 0,
                row.get("first_text") or "", reg.get("created_at")))
        # 注册了但尚未抽取完成的文档（Neo4j 还没有 Doc 节点）
        for rid, meta in registered.items():
            if rid in seen_ids:
                continue
            docs_info.append(_entry(rid, meta["filename"], user_id, 0, "", meta["created_at"]))

        docs_info.sort(key=lambda d: (d["created_at"] is None, d["created_at"] or ""))
        logger.info(f"【知识库】获取用户 {user_id} 的知识库文档，共 {len(docs_info)} 个文件")
        return docs_info

    async def get_document_detail(self, user_id: str, filename: str):
        """获取文档详情：完整内容、图片列表与切片明细（Neo4j）。"""
        try:
            records = await self._query_neo4j(
                "MATCH (d:Doc) WHERE d.user_id = $uid AND d.filename = $filename "
                "OPTIONAL MATCH (c:Chunk {kind: 'doc'}) "
                "WHERE c.source_id = d.id AND c.user_id = d.user_id "
                "RETURN d, c ORDER BY c.chunk_index",
                {"uid": user_id, "filename": filename},
            )
        except Exception as e:
            logger.error(f"【知识库】获取文档详情失败 {filename}: {e}")
            raise

        if not records:
            return None

        doc_node = records[0]["d"]
        doc_md5 = doc_node["id"]
        chunks = []
        all_images: set[str] = set()
        full_content: list[str] = []
        for index, row in enumerate(records):
            chunk = row["c"]
            if chunk is None:
                continue
            text = chunk.get("text") or ""
            full_content.append(text)
            chunk_images = self._chunk_image_urls(doc_md5, chunk.get("image_paths"))
            all_images.update(chunk_images)
            chunks.append({
                "chunk_id": chunk["id"],
                "index": chunk["chunk_index"] if chunk.get("chunk_index") is not None else index,
                "content": text,
                "page": chunk.get("page"),
                "images": chunk_images,
            })

        return {
            "id": doc_md5,
            "filename": filename,
            "user_id": doc_node.get("user_id"),
            "chunk_count": len(chunks),
            "content": "\n".join(full_content),
            "images": sorted(all_images),
            "md5": doc_md5,
            "created_at": None,
            "chunks": chunks,
        }

    async def get_document_chunks(self, user_id: str, filename: str):
        """获取文档的所有切片信息（Neo4j）。"""
        try:
            records = await self._query_neo4j(
                "MATCH (d:Doc) WHERE d.user_id = $uid AND d.filename = $filename "
                "OPTIONAL MATCH (c:Chunk {kind: 'doc'}) "
                "WHERE c.source_id = d.id AND c.user_id = d.user_id "
                "RETURN d, c ORDER BY c.chunk_index",
                {"uid": user_id, "filename": filename},
            )
        except Exception as e:
            logger.error(f"【知识库】获取文档切片失败 {filename}: {e}")
            raise

        chunks = []
        for index, row in enumerate(records):
            chunk = row["c"]
            if chunk is None:
                continue
            doc_md5 = row["d"]["id"]
            chunks.append({
                "chunk_id": chunk["id"],
                "index": chunk["chunk_index"] if chunk.get("chunk_index") is not None else index,
                "content": chunk.get("text") or "",
                "metadata": {"md5": doc_md5, "page": chunk.get("page"),
                             "user_id": row["d"].get("user_id")},
                "images": self._chunk_image_urls(doc_md5, chunk.get("image_paths")),
            })

        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }

    # ---- 删除（md5 记录 + 图片 + Neo4j 图谱联动） ----
    async def delete_user_documents(self, user_id: str):
        """删除指定用户的所有文档（包括MD5记录）"""
        try:
            await self.delete_user_md5(user_id, delete_documents=True)
        except Exception as e:
            logger.error(f"【知识库】删除用户 {user_id} 的文档时出错: {e}")
            raise

    async def delete_user_md5(self, user_id: str, delete_documents: bool = True):
        """检索内容清理成功后再删除兼容记录；连接故障应提示重试。"""
        if delete_documents:
            from app.graph.services.graph_service import cleanup_all_docs_graph
            try:
                await cleanup_all_docs_graph(user_id, strict=True)
            except Exception as exc:
                from fastapi import HTTPException
                raise HTTPException(503, "检索存储清理失败，文档尚未全部删除，请稍后重试") from exc
            delete_user_all_images(user_id)
        await self.md5_store.delete_user_md5(user_id)

    async def delete_by_filename(self, user_id: str, filename: str, delete_documents: bool = True):
        """清理全部同名历史来源，兼容 MD5 文件缺失和历史重复注册。"""
        records = await self.get_all_md5_records(user_id)
        ids = {r["md5"] for r in records
               if r.get("filename", r.get("original_filename")) == filename}
        if delete_documents:
            from app.graph.services.graph_service import cleanup_doc_graph, cleanup_doc_graph_by_filename
            try:
                deleted = set(await cleanup_doc_graph_by_filename(user_id, filename, strict=True))
                for doc_id in ids - deleted:
                    await cleanup_doc_graph(user_id, doc_id, strict=True)
                ids.update(deleted)
            except Exception as exc:
                from fastapi import HTTPException
                raise HTTPException(503, "文档清理失败，请稍后重试删除") from exc
            for doc_id in ids:
                delete_image_directory(user_id, doc_id)
        await self.md5_store.delete_by_filename(user_id, filename)
        return bool(ids)

    async def delete_single_md5(self, user_id: str, md5_to_delete: str, delete_documents: bool = True):
        exists = await self.get_md5_info(user_id, md5_to_delete) is not None
        if delete_documents:
            from app.graph.services.graph_service import cleanup_doc_graph
            try:
                await cleanup_doc_graph(user_id, md5_to_delete, strict=True)
            except Exception as exc:
                from fastapi import HTTPException
                raise HTTPException(503, "文档清理失败，请稍后重试删除") from exc
            delete_image_directory(user_id, md5_to_delete)
        await self.md5_store.delete_single_md5(user_id, md5_to_delete)
        return exists

    # ---- MD5 记录（上传去重） ----
    async def check_md5_hex(self, md5_for_check: str, user_id: str = None) -> bool:
        return await self.md5_store.check_md5_hex(md5_for_check, user_id)

    async def save_md5_hex(self, md5_hex: str, filename: str = None, original_filename: str = None, user_id: str = None):
        await self.md5_store.save_md5_hex(md5_hex, filename, original_filename, user_id)

    def save_md5_hex_sync(self, md5_hex: str, filename: str = None, original_filename: str = None, user_id: str = None):
        self.md5_store.save_md5_hex_sync(md5_hex, filename, original_filename, user_id)

    async def get_md5_info(self, user_id: str, md5_value: str):
        return next((r for r in await self.get_all_md5_records(user_id)
                     if r["md5"] == md5_value), None)

    async def get_all_md5_records(self, user_id: str):
        """优先展示持久化注册行；合并旧文件记录以兼容历史数据。"""
        from sqlalchemy import select

        from app.db.db_config import AsyncSessionLocal
        from app.models.graph import GraphDoc

        records = {r["md5"]: r for r in await self.md5_store.get_all_md5_records(user_id)}
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(select(GraphDoc.id, GraphDoc.filename, GraphDoc.created_at).where(
                GraphDoc.user_id == user_id))).all()
        for doc_id, filename, created_at in rows:
            records[doc_id] = {"md5": doc_id, "filename": filename, "original_filename": filename,
                               "upload_time": created_at.isoformat() if created_at else None}
        return list(records.values())

    # ---- 文档加载与切片（上传管线） ----
    async def get_file_document(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        return await self.document_processor.get_file_document(read_path, md5, user_id)

    def get_file_document_sync(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        return self.document_processor.get_file_document_sync(read_path, md5, user_id)

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        return self.document_processor.split_documents_sync(documents)

    async def get_document(self, files: list = None, user_id: str = None, progress_callback=None):
        """上传管线：加载 → 切片 → 保存 md5 → 入队图谱构建任务（chunk/向量由 worker 写 Neo4j）。"""
        await self.document_processor.get_document(files, user_id, progress_callback)


if __name__ == '__main__':
    async def main():
        store = VectorStoreService()
        docs = await store.get_user_documents()
        logger.debug(f"知识库文档数量: {len(docs)}")

    asyncio.run(main())
