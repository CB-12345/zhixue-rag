import asyncio

import httpx
import pytest

from app.core.settings import settings
from app.rag.agentic_rag.reranker import EvidenceReranker
from app.rag.agentic_rag.schemas import Evidence
from app.rag.agentic_rag.service import AgenticRagService
from app.rag.reorder_service import ReorderService
from tests.rag.test_agentic_rag_service import (
    FakeEvaluator,
    FakeLocalRetriever,
    FakePlanner,
    FakeWebSearchClient,
    _answerability,
    _plan,
)


def evidence(index, **updates):
    return Evidence(id=f"chunk-{index}", source="knowledge_base", title=f"资料{index}", content=f"正文{index}",
                    score=.01 * (index + 1), metadata={"source_id": f"doc-{index}", "page": index + 1,
                                                        "chunk_index": index, "vector_score": .8}, **updates)


def adapter(handler, provider="compatible"):
    transport = httpx.MockTransport(handler)
    service = ReorderService(http_client_factory=lambda **kw: httpx.AsyncClient(transport=transport, **kw))
    service.api_base_url, service.api_key, service.model = "https://rerank.test/v1", "test", "test-model"
    service.enabled = True
    service.provider = provider
    return EvidenceReranker(service)


def ascending_scores(request):
    import json

    payload = json.loads(request.content)
    size = len(payload.get("input", payload)["documents"])
    # API 返回顺序与输入相反；索引必须决定来源归属。
    output = {"results": [{"index": i, "relevance_score": i / size} for i in reversed(range(size))]}
    return httpx.Response(200, json={"output": output} if "input" in payload else output)


@pytest.mark.parametrize("provider", ["compatible", "dashscope"])
async def test_rerank_preserves_provenance_original_score_and_input(provider):
    items = [evidence(i) for i in range(12)]
    result = await adapter(ascending_scores, provider).rank("问题", items, limit=8)
    assert result.status == "applied" and result.candidate_count == 12
    assert [item.id for item in result.evidences] == [item.id for item in reversed(items[4:])]
    first = result.evidences[0]
    assert first.metadata["source_id"] == "doc-11" and first.metadata["page"] == 12
    assert first.metadata["chunk_index"] == 11 and first.metadata["vector_score"] == .8
    assert first.score == items[11].score == first.metadata["retrieval_score"]
    assert first.metadata["rerank_score"] == 11 / 12
    assert "rerank_score" not in items[11].metadata


async def test_identical_rendered_text_does_not_mix_sources():
    a = evidence(0).model_copy(update={"title": "A", "content": "B\nC"})
    b = evidence(1).model_copy(update={"title": "A\nB", "content": "C"})
    assert f"{a.title}\n{a.content}" == f"{b.title}\n{b.content}"
    result = await adapter(ascending_scores).rank("问题", [a, b])
    assert [item.metadata["source_id"] for item in result.evidences] == ["doc-1", "doc-0"]


@pytest.mark.parametrize("failure", ["timeout", "invalid_response", "error", "not_configured", "disabled"])
@pytest.mark.parametrize("provider", ["compatible", "dashscope"])
async def test_service_failure_keeps_original_order(failure, provider):
    async def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("test")
        if failure == "invalid_response":
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": .9}]})
        if failure == "error":
            return httpx.Response(503)
        pytest.fail("未配置或关闭时不应该请求网络")

    ranker = adapter(handler, provider)
    if failure == "not_configured":
        ranker.service.api_key = ""
    if failure == "disabled":
        ranker.service.enabled = False
    items = [evidence(i) for i in range(10)]
    result = await ranker.rank("问题", items)
    assert result.status == failure and result.evidences == items[:settings.RAG_CONTEXT_K]


async def test_total_timeout_cancels_slow_provider():
    cancelled = asyncio.Event()

    async def handler(request):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    ranker = adapter(handler)
    ranker.service.timeout = .02
    items = [evidence(0), evidence(1)]
    result = await ranker.rank("问题", items)
    assert result.status == "timeout" and result.evidences == items
    assert cancelled.is_set()


@pytest.mark.parametrize("provider", ["compatible", "dashscope"])
@pytest.mark.parametrize("context_k", [3, 5, 8])
async def test_pipeline_expands_candidates_reranks_before_evaluation_and_context(monkeypatch, provider, context_k):
    monkeypatch.setattr(settings, "RERANKER_CANDIDATE_K", 20)
    monkeypatch.setattr(settings, "RAG_CONTEXT_K", context_k)
    plan = _plan()
    original_top_k = plan.steps[0].top_k
    local = FakeLocalRetriever([evidence(i) for i in range(20)])
    evaluator = FakeEvaluator(_answerability())
    events = []
    service = AgenticRagService(planner=FakePlanner(plan), local_retriever=local, evaluator=evaluator,
                               web_search_client=FakeWebSearchClient(), reranker=adapter(ascending_scores, provider))
    result = await service.run("当前问题", "u-1", events.append)
    executed = next(call for call in local.calls if call["steps"])
    assert executed["steps"][0].top_k == 20 and plan.steps[0].top_k == original_top_k
    assert result.evidences[0].id == "chunk-19"
    assert len(result.evidences) == context_k
    assert evaluator.calls[0]["evidences"] == result.evidences
    assert "正文19" in result.context and "正文0\n" not in result.context
    assert result.rerank["status"] == "applied" and result.rerank["candidate_count"] == 20
    stages = [event["stage"] for event in events]
    assert stages.index("reorder") < stages.index("answerability")


async def test_pipeline_provider_failure_still_builds_answer_context():
    items = [evidence(i) for i in range(12)]
    service = AgenticRagService(planner=FakePlanner(_plan()), local_retriever=FakeLocalRetriever(items),
                               evaluator=FakeEvaluator(_answerability()), web_search_client=FakeWebSearchClient(),
                               reranker=adapter(lambda request: httpx.Response(500)))
    result = await service.run("问题", "u-1")
    assert result.evidences == items[:settings.RAG_CONTEXT_K] and result.rerank["status"] == "error"
    assert "正文0" in result.context


@pytest.mark.parametrize("context_k", [3, 5, 8])
async def test_expanded_local_pool_leaves_room_for_web_evidence(monkeypatch, context_k):
    monkeypatch.setattr(settings, "RAG_CONTEXT_K", context_k)
    web = Evidence(id="web-1", source="web", title="补充", content="独有网络证据", url="https://example.org")
    service = AgenticRagService(planner=FakePlanner(_plan()), local_retriever=FakeLocalRetriever([evidence(i) for i in range(20)]),
                               evaluator=FakeEvaluator([_answerability(answerable=False), _answerability()]),
                               web_search_client=FakeWebSearchClient([web]), reranker=adapter(ascending_scores))
    result = await service.run("问题", "u-1")
    assert len(result.evidences) == min(context_k, 7) + 1 and result.evidences[-1] == web
    assert result.used_web and "独有网络证据" in result.context
