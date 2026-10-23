"""面向用户的检索依据快照；只暴露可核对内容，不透传内部 metadata。"""
from urllib.parse import urlparse


def source_page(value) -> int | None:
    """仅展示可靠的正整数物理页码，不猜测缺失或旧的零页码。"""
    return value if type(value) is int and value > 0 else None


def public_sources(evidences) -> list[dict]:
    sources = []
    for index, evidence in enumerate(evidences[:8], 1):
        metadata = evidence.metadata or {}
        url = evidence.url
        if url and urlparse(url).scheme not in {"http", "https"}:
            url = None
        sources.append({
            "number": index,
            "id": evidence.id,
            "source": evidence.source,
            "title": evidence.title,
            "content": evidence.content[:1600],
            "source_id": metadata.get("source_id"),
            "page": source_page(metadata.get("page")),
            "chunk_index": metadata.get("chunk_index"),
            "url": url,
        })
    return sources
