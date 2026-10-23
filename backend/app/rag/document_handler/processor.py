"""文档处理器：多格式加载 + 切片（多模态 PDF 保留页码/图片元数据）。

上传管线入口 get_document 复用 KnowledgeService：加载 → 切片 → 原子登记文档与任务。
基础索引 worker 独立写入 chunk/向量；MD5 文件仅为兼容记录。
"""
import os

from langchain_core.documents import Document

from app.rag.text_spliter import AsyncTextSplitter
from app.utils.config import document_config
from app.utils.file_handler import (
    listdir_allowed_type,
    markdown_loader,
    markdown_loader_sync,
    pdf_loader,
    pdf_loader_sync,
    ppt_loader,
    ppt_loader_sync,
    txt_loader,
    txt_loader_sync,
    word_loader,
    word_loader_sync,
)
from app.utils.pdf_multimodal_loader import pdf_multimodal_loader, pdf_multimodal_loader_sync


class DocumentProcessor:
    """文档处理器：加载与切片（chunk/向量写入由图谱构建 worker 落 Neo4j）"""

    def __init__(self, md5_store, embed_model=None):
        self.md5_store = md5_store
        self.spliter = AsyncTextSplitter(
            chunk_size=document_config['chunk_size'],
            chunk_overlap=document_config['chunk_overlap'],
            separators=document_config['separators'],
            embedding_model=embed_model
        )

    async def get_file_document(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        """异步加载文件"""
        if read_path.endswith('.txt'):
            return await txt_loader(read_path)
        elif read_path.endswith('.pdf'):
            # 优先使用多模态加载器（提取图片+视觉描述），仅当提供了md5和user_id时才启用；
            # 这两个参数用于定位图片的存储路径 data/extracted_images/{user_id}/{md5}/
            if md5 and user_id:
                return await pdf_multimodal_loader(read_path, md5, user_id)
            # 回退到纯文本加载器（仅提取文字，无图片）
            return await pdf_loader(read_path)
        elif read_path.endswith('.md'):
            return await markdown_loader(read_path)
        elif read_path.endswith('.pptx'):
            return await ppt_loader(read_path)
        elif read_path.endswith('.docx'):
            return await word_loader(read_path)
        else:
            return []

    def get_file_document_sync(self, read_path: str, md5: str = None, user_id: str = None) -> list[Document]:
        """同步加载文件（用于多线程场景）"""
        if read_path.endswith('.txt'):
            return txt_loader_sync(read_path)
        elif read_path.endswith('.pdf'):
            if md5 and user_id:
                return pdf_multimodal_loader_sync(read_path, md5, user_id)
            return pdf_loader_sync(read_path)
        elif read_path.endswith('.md'):
            return markdown_loader_sync(read_path)
        elif read_path.endswith('.pptx'):
            return ppt_loader_sync(read_path)
        elif read_path.endswith('.docx'):
            return word_loader_sync(read_path)
        else:
            return []

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        """同步分割文档（用于多线程场景）"""
        return self.spliter.split_documents_sync(documents)

    async def get_document(self, files: list = None, user_id: str = None, progress_callback=None):
        """复用上传登记管线，避免普通上传漏记任务或依赖过时的 MD5 标记。"""
        from contextlib import ExitStack

        from fastapi import UploadFile

        from app.router.knowledge_service import KnowledgeService

        with ExitStack() as stack:
            if not files:
                paths = await listdir_allowed_type(
                    document_config['data_path'], tuple(document_config['allow_knowledge_file_types']))
                files = [UploadFile(filename=os.path.basename(path),
                                    file=stack.enter_context(open(path, "rb"))) for path in paths]
            await KnowledgeService().consume_upload(files, user_id, progress_callback)
