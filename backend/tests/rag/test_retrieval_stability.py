import asyncio

import pytest

from app.core.settings import settings
from app.rag.agentic_rag.schemas import Evidence, RetrievalPlan, RetrievalStep
from app.rag.agentic_rag.service import AgenticRagService
from tests.rag.test_agentic_rag_service import FakeEvaluator, FakePlanner, FakeWebSearchClient, _answerability, _plan


def service_with(retriever):
    plan = _plan()
    plan.steps.append(RetrievalStep(tool="search_graph", query="fixture"))
    return AgenticRagService(planner=FakePlanner(plan), local_retriever=retriever,
        evaluator=FakeEvaluator(_answerability()), web_search_client=FakeWebSearchClient())


async def test_slow_optional_graph_keeps_text_evidence_and_cancels_graph(monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_RETRIEVAL_TIMEOUT_SECONDS", .03, raising=False)
    cancelled = asyncio.Event()
    fact = Evidence(id="text", source="knowledge_base", title="fixture.txt", content="负责人林乔")

    class Retriever:
        async def search(self, uid, steps):
            if steps and steps[0].tool == "search_graph":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return [fact]

    result = await asyncio.wait_for(service_with(Retriever()).run("查询负责人", "u1"), .5)
    assert result.evidences == [fact] and not result.refusal_message
    assert cancelled.is_set()


async def test_cancelling_pipeline_drains_both_parallel_retrieval_tasks():
    graph_started, text_started = asyncio.Event(), asyncio.Event()
    cancelled, tasks = set(), []

    class Retriever:
        async def search(self, uid, steps):
            kind = "graph" if steps and steps[0].tool == "search_graph" else "text"
            tasks.append(asyncio.current_task())
            (graph_started if kind == "graph" else text_started).set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(kind)

    task = asyncio.create_task(service_with(Retriever()).run("查询负责人", "u1"))
    try:
        await asyncio.wait_for(asyncio.gather(graph_started.wait(), text_started.wait()), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled == {"graph", "text"}
        assert all(child.done() for child in tasks)
    finally:
        for child in [task, *tasks]:
            child.cancel()
        await asyncio.gather(task, *tasks, return_exceptions=True)


async def test_graph_only_timeout_cannot_bypass_missing_evidence_gate(monkeypatch):
    monkeypatch.setattr(settings, "GRAPH_RETRIEVAL_TIMEOUT_SECONDS", .01)

    class Retriever:
        async def search(self, uid, steps):
            if steps:
                await asyncio.Event().wait()
            return []

    service = service_with(Retriever())
    service.planner = FakePlanner(RetrievalPlan(need_retrieval=True,
        steps=[RetrievalStep(tool="search_graph", query="fixture")]))
    result = await service.run("查询负责人", "u1")
    assert result.refusal_message and not result.evidences and not result.context
