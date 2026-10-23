from app.rag.agentic_rag.evidence import format_evidence_context, merge_evidence, prepare_evidence_context
from app.rag.agentic_rag.schemas import Evidence


def test_merge_evidence_dedupes_by_source_id_before_content():
    evidences = [
        Evidence(id="1", source="note", title="First", content="Same local fact"),
        Evidence(id="1", source="note", title="Duplicate ID", content="Different text"),
        Evidence(id="2", source="knowledge_base", title="KB", content="  Same   local fact  "),
        Evidence(id="3", source="web", title="Web", content="Fresh external fact"),
    ]

    merged = merge_evidence(evidences)

    assert [(item.source, item.id, item.title) for item in merged] == [
        ("note", "1", "First"),
        ("web", "3", "Web"),
    ]


def test_merge_evidence_applies_limit_after_dedupe():
    evidences = [
        Evidence(id=str(i), source="note", title=f"Note {i}", content=f"Content {i}")
        for i in range(3)
    ]

    merged = merge_evidence(evidences, limit=2)

    assert [item.id for item in merged] == ["0", "1"]


def test_format_evidence_context_uses_chinese_source_labels_and_budget():
    evidences = [
        Evidence(id="n1", source="note", title="读书笔记", content="笔记正文"),
        Evidence(id="k1", source="knowledge_base", title="Manual.pdf", content="知识库正文"),
        Evidence(id="w1", source="web", title="News", content="外部正文", url="https://example.com/news"),
    ]

    context = format_evidence_context(evidences, max_chars=80)

    assert "[1] 来源：笔记《读书笔记》" in context
    assert "笔记正文" in context
    assert "[2] 来源：知识库《Manual.pdf》" in context
    assert "知识库正文" in context
    assert "来源：外部搜索" not in context
    assert len(context) <= 80


def test_format_evidence_context_includes_web_url_when_present():
    evidences = [
        Evidence(id="w1", source="web", title="Search Result", content="搜索摘要", url="https://example.com/a"),
    ]

    context = format_evidence_context(evidences)

    assert "[1] 来源：外部搜索《Search Result》" in context
    assert "URL：https://example.com/a" in context
    assert "搜索摘要" in context


def test_oversized_first_chunk_is_clipped_and_omitted_sources_are_not_returned():
    original = Evidence(id="large", source="knowledge_base", title="大文档", content="正文内容" * 3000,
                        metadata={"source_id": "doc1", "page": 7})
    later = Evidence(id="later", source="note", title="后续资料", content="没有进入模型的内容")
    prepared = prepare_evidence_context([original, later], max_chars=500)
    assert len(prepared.text) == 500 and len(prepared.evidences) == 1
    assert prepared.evidences[0].metadata == {"source_id": "doc1", "page": 7, "context_truncated": True}
    assert prepared.text.endswith(prepared.evidences[0].content)
    assert "后续资料" not in prepared.text and "context_truncated" not in original.metadata
    assert len(original.content) == 12000


def test_empty_chunks_do_not_receive_citation_numbers():
    empty = Evidence(id="empty", source="note", title="空", content=" ")
    fact = Evidence(id="fact", source="note", title="真实资料", content="有效事实")
    prepared = prepare_evidence_context([empty, fact])
    assert prepared.evidences == [fact] and prepared.text.startswith("[1] 来源")
