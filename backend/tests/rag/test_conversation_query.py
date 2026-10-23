"""多轮指代补全与证据门控：历史用于理解对象，不能代替当前检索证据。"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.rag.agentic_rag import conversation
from app.rag.agentic_rag.conversation import ConversationQueryResolver, QueryResolution
from app.rag.agentic_rag.schemas import AnswerabilityResult, Evidence, RetrievalPlan
from app.rag.agentic_rag.service import AgenticRagService
from tests.fakes import make_fake_chat_model

HISTORY = [("仅依据我的文档，晨星计划的目标是什么？", "目标是验证混合检索。")]
RESOLVED = "晨星计划的负责人是谁？"


def decision(**changes):
    result = {"status": "resolved", "standalone_query": RESOLVED,
              "references": [{"turn": 1, "role": "user", "quote": "晨星计划"}]}
    result.update(changes)
    return json.dumps(result, ensure_ascii=False)


async def test_follow_up_copies_verified_identifier_and_preserves_local_scope():
    resolver = ConversationQueryResolver(make_fake_chat_model([decision()]))
    result = await resolver.resolve("它的负责人呢？", HISTORY)
    assert result.status == "resolved" and result.query == RESOLVED and result.local_only
    assert result.method == "model"


@pytest.mark.parametrize("query, expected", [
    ("它的负责人呢？", "学习记录 HYB-2048的负责人呢？"),
    ("这条记录的负责人和电话呢？不要推测。", "学习记录 HYB-2048的负责人和电话呢？不要推测。"),
])
async def test_single_identifier_follow_up_skips_model_and_preserves_requirements(query, expected):
    model = SimpleNamespace(ainvoke=AsyncMock())
    history = [("仅依据我的资料，学习记录 HYB-2048 的实验候选数量是多少？",
                "HYB-2048 的实验候选数量是 6。")]
    result = await ConversationQueryResolver(model).resolve(query, history)
    assert result.query == expected and result.status == "resolved"
    assert result.method == "identifier" and result.local_only
    model.ainvoke.assert_not_called()


@pytest.mark.parametrize("query, history", [
    ("它的负责人呢？", [("比较学习记录 HYB-2048 与另一个项目。", "两项实验的目的不同。")]),
    ("它的负责人呢？", [("学习记录 HYB-2048 的目标是什么？", "另见 NOTE-4096 的介绍。")]),
    ("它的负责人呢？", [("HYB-2048 的目标是什么？", "实验混合检索。")]),
    ("它的负责人呢？", [("学习记录 HYB-2048 的目标是什么？", "实验混合检索。"),
                          ("晨星计划呢？", "重点是学习进度。")]),
    ("他的电话号码呢？", [("学习记录 HYB-2048 的负责人是谁？", "负责人是周予安。")]),
    ("它与 NOTE-4096 的区别呢？", [("学习记录 HYB-2048 的目标是什么？", "实验混合检索。")]),
])
async def test_identifier_shortcut_does_not_guess_subject(query, history):
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(
        content=decision(status="clarify", standalone_query="", references=[]))))
    result = await ConversationQueryResolver(model).resolve(query, history)
    assert result.status == "clarify" and result.method == "model"
    model.ainvoke.assert_awaited_once()


async def test_independent_question_and_greeting_skip_extra_model_call():
    model = SimpleNamespace(ainvoke=AsyncMock())
    for query in ["HYB-2048 的负责人是谁？", "你好", "讲讲 BM25"]:
        result = await ConversationQueryResolver(model).resolve(query, HISTORY)
        assert result.query == query and result.status == "unchanged"
    model.ainvoke.assert_not_called()


async def test_follow_up_without_history_asks_for_subject_without_model_call():
    model = SimpleNamespace(ainvoke=AsyncMock())
    result = await ConversationQueryResolver(model).resolve("负责人呢？")
    assert result.status == "clarify" and result.clarification
    model.ainvoke.assert_not_called()


@pytest.mark.parametrize("payload", [
    "bad json",
    decision(references=[]),
    decision(references=[{"turn": 2, "role": "user", "quote": "晨星计划"}]),
    decision(references=[{"turn": 1, "role": "user", "quote": "FAKE-9999"}]),
    decision(standalone_query="另一个项目的负责人是谁？"),
    decision(status="clarify"),
    decision(status="resolved", standalone_query=""),
])
async def test_invalid_or_invented_referent_cannot_start_retrieval(payload):
    result = await ConversationQueryResolver(make_fake_chat_model([payload])).resolve("它的负责人呢？", HISTORY)
    assert result.status == "unavailable" and result.clarification


async def test_ambiguous_subject_is_clarified():
    payload = decision(status="clarify", standalone_query="", references=[])
    result = await ConversationQueryResolver(make_fake_chat_model([payload])).resolve(
        "它的负责人呢？", [("比较 HYB-2048 与 RERANK-4096", "两项实验目的不同。")])
    assert result.status == "clarify" and "明确" in result.clarification


async def test_timeout_asks_for_explicit_identifier(monkeypatch):
    async def slow(_):
        await asyncio.Event().wait()
    monkeypatch.setattr(conversation.settings, "QUERY_REWRITE_TIMEOUT_SECONDS", 0.1)
    result = await ConversationQueryResolver(SimpleNamespace(ainvoke=slow)).resolve("它的负责人呢？", HISTORY)
    assert result.status == "unavailable"


async def test_unavailable_model_asks_for_explicit_identifier(monkeypatch):
    monkeypatch.setattr(conversation, "_resolve_planner_chat_model", lambda: None)
    result = await ConversationQueryResolver().resolve("它的负责人呢？", HISTORY)
    assert result.status == "unavailable"


async def test_history_is_bounded_and_encoded_as_data():
    captured = []
    async def invoke(prompt):
        captured.append(json.loads(prompt.rsplit("Input JSON:\n", 1)[1]))
        return SimpleNamespace(content=decision(status="clarify", standalone_query="", references=[]))
    history = [("oldest-secret", "not-needed")] + [("u" * 1000, "a" * 2000)] * 4
    await ConversationQueryResolver(SimpleNamespace(ainvoke=invoke)).resolve("它的负责人呢？", history)
    assert len(captured[0]["history"]) == 3
    assert all(len(t["user"]) == 600 and len(t["assistant"]) == 900 for t in captured[0]["history"])
    assert "oldest-secret" not in json.dumps(captured)


async def test_explicit_web_request_can_change_previous_local_scope():
    result = await ConversationQueryResolver(make_fake_chat_model([decision()])).resolve(
        "联网核对它的负责人", HISTORY)
    assert result.status == "resolved" and not result.local_only


async def test_service_uses_resolved_subject_but_checks_original_requested_facts():
    seen = {}
    resolver = SimpleNamespace(resolve=AsyncMock(return_value=QueryResolution(RESOLVED, "resolved", True)))
    planner = SimpleNamespace(plan=AsyncMock(return_value=RetrievalPlan(need_retrieval=False, steps=[])))
    evidence = Evidence(id="chunk", source="knowledge_base", title="实验说明", content="晨星计划的负责人是周予安。")
    retriever = SimpleNamespace(search=AsyncMock(return_value=[evidence]))
    async def evaluate(query, evidences):
        seen["query"], seen["evidences"] = query, evidences
        return AnswerabilityResult(answerable=False, confidence=0.9, reason="没有电话", web_queries=["phone"])
    evaluator = SimpleNamespace(evaluate=evaluate)
    web = SimpleNamespace(search=AsyncMock(return_value=[]))
    result = await AgenticRagService(planner=planner, local_retriever=retriever, evaluator=evaluator,
                                   web_search_client=web, query_resolver=resolver).run(
        "它的负责人和电话呢？不要推测。", "owner", history=HISTORY)
    planner.plan.assert_awaited_once_with(RESOLVED)
    assert "电话" in seen["query"] and "不要推测" in seen["query"] and RESOLVED in seen["query"]
    assert retriever.search.await_args_list[0].args[1][0].query == RESOLVED
    assert seen["evidences"] == [evidence]  # 没有把历史回答混入证据。
    web.search.assert_not_called()
    assert result.refusal_message and result.evidences == []


async def test_clarification_does_not_invoke_planner_retriever_or_evaluator():
    planner = SimpleNamespace(plan=AsyncMock())
    retriever = SimpleNamespace(search=AsyncMock())
    evaluator = SimpleNamespace(evaluate=AsyncMock())
    result = await AgenticRagService(planner=planner, local_retriever=retriever, evaluator=evaluator).run(
        "它的负责人呢？", "owner", history=[])
    assert result.refusal_message and not result.plan.need_retrieval
    planner.plan.assert_not_called()
    retriever.search.assert_not_called()
    evaluator.evaluate.assert_not_called()
