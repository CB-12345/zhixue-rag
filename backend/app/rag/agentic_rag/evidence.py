import re
from dataclasses import dataclass

from app.rag.agentic_rag.schemas import Evidence
from app.rag.agentic_rag.sources import source_page

_SOURCE_LABELS = {
    "note": "笔记",
    "knowledge_base": "知识库",
    "graph": "知识图谱",
    "web": "外部搜索",
}


def _normalized_content(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip().lower()


def merge_evidence(evidences: list[Evidence], limit: int = 8) -> list[Evidence]:
    merged: list[Evidence] = []
    seen_source_ids: set[tuple[str, str]] = set()
    seen_content: set[str] = set()

    for evidence in evidences:
        source_id = (evidence.source, evidence.id)
        if source_id in seen_source_ids:
            continue

        content_key = _normalized_content(evidence.content)
        if content_key and content_key in seen_content:
            continue

        seen_source_ids.add(source_id)
        if content_key:
            seen_content.add(content_key)
        merged.append(evidence)

        if len(merged) >= limit:
            break

    return merged


@dataclass
class EvidenceContext:
    text: str
    evidences: list[Evidence]


def prepare_evidence_context(evidences: list[Evidence], max_chars: int = 6000) -> EvidenceContext:
    """同步裁剪上下文与来源快照，避免展示未实际送入模型的资料。"""
    parts: list[str] = []
    selected: list[Evidence] = []
    used = 0

    for evidence in evidences:
        if not evidence.content.strip():
            continue
        index = len(selected) + 1
        label = _SOURCE_LABELS[evidence.source]
        header = f"[{index}] 来源：{label}《{evidence.title}》"
        page = source_page(evidence.metadata.get("page"))
        if page is not None:
            header += f"（第 {page} 页）"
        lines = [header]
        if evidence.url:
            lines.append(f"URL：{evidence.url}")
        prefix = "\n".join(lines) + "\n"
        separator = "\n\n" if parts else ""
        remaining = max_chars - used - len(separator) - len(prefix)
        if remaining <= 0:
            break
        content = evidence.content[:remaining]
        if not content.strip():
            break
        if content != evidence.content:
            evidence = evidence.model_copy(update={"content": content, "metadata": {
                **evidence.metadata, "context_truncated": True}})
        block = prefix + content
        parts.append(block)
        selected.append(evidence)
        used += len(separator) + len(block)

    return EvidenceContext("\n\n".join(parts), selected)


def format_evidence_context(evidences: list[Evidence], max_chars: int = 6000) -> str:
    return prepare_evidence_context(evidences, max_chars).text
