import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from app.rag.agentic_rag import evaluator as evaluator_module
from app.rag.agentic_rag.evaluator import AnswerabilityEvaluator
from app.rag.agentic_rag.schemas import Evidence


class FakeChatModel:
    """ainvoke 返回预设文本（或抛预设异常），记录收到的 prompt。"""

    def __init__(self, content):
        self.content = content
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.content, Exception):
            raise self.content
        return type("Response", (), {"content": self.content})()


def _evidence(content="李白的代表作有《蜀道难》《将进酒》。", score=0.8):
    return Evidence(id="ev-1", source="knowledge_base", title="Doc", content=content, score=score)


def _llm_json(**overrides) -> str:
    payload = {"answerable": True, "confidence": 0.8, "reason": "evidence relevant", "web_queries": [],
               "supporting_quotes": [{"evidence_id": "ev-1", "quote": "李白的代表作有《蜀道难》《将进酒》。"}]}
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_empty_evidence_short_circuits_without_llm():
    model = FakeChatModel(content="{}")
    evaluator = AnswerabilityEvaluator(chat_model=model)

    result = await evaluator.evaluate("What is agentic RAG?", [])

    assert result.answerable is False
    assert result.confidence == 0.0
    assert result.web_queries == ["What is agentic RAG?"]
    assert model.prompts == []  # 快路径不应消耗 LLM 调用


@pytest.mark.parametrize("url,official", [
    ("https://api.deepseek.com/v1", True),
    ("https://api.example.test/v1", False),
    ("https://api.deepseek.com.example.test/v1", False),
])
async def test_short_evidence_review_does_not_change_shared_answer_model(monkeypatch, url, official):
    shared = Mock()
    response = type("Response", (), {"content": _llm_json()})()
    shared.ainvoke = AsyncMock(return_value=response)
    bound = FakeChatModel(_llm_json())
    shared.bind.return_value = bound
    monkeypatch.setattr(evaluator_module, "_resolve_shared_chat_model", lambda: shared)
    monkeypatch.setattr(evaluator_module.settings, "OPENAI_BASE_URL", url)
    result = await AnswerabilityEvaluator().evaluate("李白的代表作", [_evidence()])
    assert result.answerable and result.supporting_quotes[0].evidence_id == "ev-1"
    if official:
        shared.bind.assert_called_once_with(extra_body={"thinking": {"type": "disabled"}})
        shared.ainvoke.assert_not_called()
        assert len(bound.prompts) == 1
    else:
        shared.bind.assert_not_called()
        shared.ainvoke.assert_awaited_once()


async def test_explicitly_supplied_evaluator_keeps_its_own_request_options(monkeypatch):
    model = FakeChatModel(_llm_json())
    monkeypatch.setattr(evaluator_module.settings, "OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    result = await AnswerabilityEvaluator(chat_model=model).evaluate("李白的代表作", [_evidence()])
    assert result.answerable and len(model.prompts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["OpenAI current price news", "今年 LangChain 版本 变化"])
async def test_freshness_queries_still_require_content_verification(query):
    model = FakeChatModel(content=_llm_json(answerable=False, confidence=.2, web_queries=[query], supporting_quotes=[]))
    evaluator = AnswerabilityEvaluator(chat_model=model)

    result = await evaluator.evaluate(query, [_evidence()])

    assert result.answerable is False
    assert result.confidence < 0.5
    assert result.web_queries == [query]
    assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_llm_judges_unrelated_evidence_not_answerable():
    """白居易场景：笔记里没有该人物，top-k 噪声证据不构成"可答"。"""
    model = FakeChatModel(content=_llm_json(
        answerable=False, confidence=0.15,
        reason="Evidence is about quantum computing, not Bai Juyi.",
        web_queries=["白居易 唐代诗人"], supporting_quotes=[],
    ))
    evaluator = AnswerabilityEvaluator(chat_model=model)

    result = await evaluator.evaluate("给我讲讲白居易", [_evidence(content="量子计算使用量子比特…")])

    assert result.answerable is False
    assert result.confidence == 0.15
    assert result.web_queries == ["白居易 唐代诗人"]
    assert len(model.prompts) == 1
    assert "给我讲讲白居易" in model.prompts[0]
    assert "量子计算" in model.prompts[0]  # 证据片段进入 prompt


@pytest.mark.asyncio
async def test_llm_judges_relevant_evidence_answerable():
    model = FakeChatModel(content=_llm_json(answerable=True, confidence=0.9, web_queries=[]))
    evaluator = AnswerabilityEvaluator(chat_model=model)

    result = await evaluator.evaluate("李白的代表作有哪些", [_evidence()])

    assert result.answerable is True
    assert result.confidence == 0.9
    assert result.web_queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["false", "False", "0", "no"])
async def test_llm_string_answerable_is_coerced(raw):
    model = FakeChatModel(content=_llm_json(answerable=raw))
    evaluator = AnswerabilityEvaluator(chat_model=model)

    result = await evaluator.evaluate("Tell me about Bai Juyi", [_evidence()])

    assert result.answerable is False
    assert result.web_queries == ["Tell me about Bai Juyi"]  # 不可答但缺 web_queries 时兜底原问题


@pytest.mark.asyncio
async def test_llm_malformed_json_does_not_allow_answering():
    model = FakeChatModel(content="这不是 JSON")
    evaluator = AnswerabilityEvaluator(chat_model=model)

    result = await evaluator.evaluate("Summarize local architecture notes", [_evidence()])

    assert result.answerable is False
    assert result.status == "unavailable"
    assert result.web_queries == []


@pytest.mark.asyncio
async def test_llm_failure_does_not_allow_answering():
    model = FakeChatModel(content=RuntimeError("llm down"))
    evaluator = AnswerabilityEvaluator(chat_model=model)

    result = await evaluator.evaluate("Summarize local architecture notes", [_evidence()])

    assert result.answerable is False
    assert result.status == "unavailable"
    assert result.web_queries == []


@pytest.mark.asyncio
async def test_no_available_model_does_not_allow_answering(monkeypatch):
    monkeypatch.setattr(evaluator_module, "_resolve_shared_chat_model", lambda: None)
    evaluator = AnswerabilityEvaluator()

    result = await evaluator.evaluate("Summarize local architecture notes", [_evidence()])

    assert result.answerable is False
    assert result.status == "unavailable"
    assert result.web_queries == []


@pytest.mark.parametrize("overrides", [
    {"answerable": "maybe"}, {"answerable": 1}, {"confidence": float("nan")},
    {"confidence": float("inf")}, {"confidence": True}, {"confidence": 1.1},
    {"supporting_quotes": []}, {"supporting_quotes": [{"evidence_id": "invented", "quote": "李白"}]},
    {"supporting_quotes": [{"evidence_id": "ev-1", "quote": "杜甫的代表作是春望"}]},
    {"supporting_quotes": [{"evidence_id": "ev-1", "quote": " "}]}, {"web_queries": "query"},
])
async def test_invalid_decision_or_fabricated_quote_is_not_accepted(overrides):
    result = await AnswerabilityEvaluator(chat_model=FakeChatModel(_llm_json(**overrides))).evaluate("李白的代表作", [_evidence()])
    assert not result.answerable and result.status == "unavailable"


async def test_full_content_after_first_300_chars_is_checked():
    content = "背景说明。" * 100 + "负责人是周予安。"
    model = FakeChatModel(_llm_json(supporting_quotes=[{"evidence_id": "ev-1", "quote": "负责人是周予安。"}]))
    result = await AnswerabilityEvaluator(chat_model=model).evaluate("负责人是谁？", [_evidence(content)])
    assert result.answerable and result.supporting_quotes[0].quote == "负责人是周予安。"
    assert "负责人是周予安。" in model.prompts[0]


async def test_docx_breaks_do_not_make_valid_evidence_unavailable():
    model = FakeChatModel(_llm_json(supporting_quotes=[{'evidence_id': 'ev-1', 'quote': 'count()\n非缺失值个数'}]))
    result = await AnswerabilityEvaluator(model).evaluate('count统计什么？', [_evidence('count()<br>非缺失值个数')])
    assert result.answerable and result.status == 'evaluated'


async def test_opt_in_diagnostics_preserve_invalid_raw_quote_without_allowing_answering():
    raw = _llm_json(supporting_quotes=[{'evidence_id': 'ev-1', 'quote': '没有依据的结论'}])
    diagnostics = {'stale': 'discard'}
    result = await AnswerabilityEvaluator(FakeChatModel(raw), diagnostics=diagnostics).evaluate('代表作？', [_evidence()])
    assert not result.answerable and diagnostics['error_type'] == 'ValueError'
    assert diagnostics['raw_review'] == raw and diagnostics['status'] == 'unavailable' and 'stale' not in diagnostics


@pytest.mark.parametrize("http_status", [402, "secret-response", True])
async def test_private_provider_failure_status_does_not_expose_exception_body(http_status, monkeypatch):
    class ProviderFailure(Exception):
        status_code = http_status
        body = {"error": "secret-provider-body"}

    class FailedModel:
        async def ainvoke(self, prompt):
            raise ProviderFailure("secret-key-in-message")

    warnings = []
    monkeypatch.setattr(evaluator_module.logger, "warning", warnings.append)
    diagnostics = {}
    result = await AnswerabilityEvaluator(FailedModel(), diagnostics=diagnostics).evaluate("代表作？", [_evidence()])
    assert result.status == "unavailable" and not result.answerable
    assert diagnostics.get("http_status") == (402 if type(http_status) is int else None)
    assert "secret-" not in json.dumps(diagnostics) + str(warnings)


_REFERENCE_CASES = json.loads((Path(__file__).resolve().parents[1] / "fixtures/answerability_reference.json").read_bytes())["cases"]


@pytest.mark.parametrize("case", _REFERENCE_CASES, ids=lambda case: case["id"])
async def test_reference_repair_and_rejection_controls(case):
    evidence = [Evidence.model_validate(item) for item in case["evidence"]]
    raw = json.dumps(case["decision"], ensure_ascii=False)
    diagnostics = {}
    model = FakeChatModel(raw)
    result = await AnswerabilityEvaluator(model, diagnostics=diagnostics).evaluate(case["query"], evidence)
    assert result.answerable is case["expected_answerable"]
    assert len(model.prompts) == 1 and diagnostics["raw_review"] == raw
    if case["id"] == "a01-same-document-reference":
        assert diagnostics["quote_reference_repairs"] == [{
            "from_evidence_id": evidence[1].id, "to_evidence_id": evidence[2].id,
        }]
        assert result.supporting_quotes[3].evidence_id == evidence[2].id
        assert case["decision"]["supporting_quotes"][3]["evidence_id"] == evidence[1].id
    elif not case["expected_answerable"]:
        assert result.status == "unavailable" and not result.supporting_quotes


async def test_reference_repair_cannot_use_unsubmitted_ninth_evidence():
    wrong = _evidence("不含目标的片段").model_copy(update={"metadata": {"source_id": "doc-a"}})
    filler = [wrong.model_copy(update={"id": f"filler-{i}"}) for i in range(7)]
    ninth = wrong.model_copy(update={"id": "ninth", "content": "真实摘录，仅位于未提交的第九片。"})
    raw = _llm_json(supporting_quotes=[{"evidence_id": wrong.id, "quote": ninth.content}])
    result = await AnswerabilityEvaluator(FakeChatModel(raw)).evaluate("目标？", [wrong, *filler, ninth])
    assert not result.answerable and result.status == "unavailable"


async def test_reference_repair_requires_document_identity_metadata():
    wrong, actual = _evidence("没有目标"), _evidence("真实摘录").model_copy(update={"id": "ev-2"})
    raw = _llm_json(supporting_quotes=[{"evidence_id": wrong.id, "quote": actual.content}])
    result = await AnswerabilityEvaluator(FakeChatModel(raw)).evaluate("目标？", [wrong, actual])
    assert not result.answerable and result.status == "unavailable"


async def test_timeout_cancels_verification_and_does_not_allow_answering(monkeypatch):
    cancelled = asyncio.Event()

    class SlowModel:
        async def ainvoke(self, prompt):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    monkeypatch.setattr(evaluator_module.settings, "ANSWERABILITY_TIMEOUT_SECONDS", .01)
    result = await AnswerabilityEvaluator(chat_model=SlowModel()).evaluate("代表作", [_evidence()])
    assert not result.answerable and result.status == "unavailable" and cancelled.is_set()


async def test_local_document_version_does_not_require_external_freshness():
    model = FakeChatModel(_llm_json())
    result = await AnswerabilityEvaluator(chat_model=model).evaluate("只依据我的资料说明版本规范", [_evidence()])
    assert result.answerable and len(model.prompts) == 1


async def test_revision_workflow_is_not_mistaken_for_live_external_version():
    content = "新版本向量计算失败时，必须保留原有可查询资料。"
    model = FakeChatModel(_llm_json(supporting_quotes=[{"evidence_id": "ev-1", "quote": content}]))
    result = await AnswerabilityEvaluator(chat_model=model).evaluate("新版本的向量计算失败后，原有可查询资料应该怎样处理？", [_evidence(content)])
    assert result.answerable and len(model.prompts) == 1


async def test_fresh_web_evidence_is_evaluated_instead_of_repeating_freshness_rule():
    model = FakeChatModel(_llm_json())
    web = _evidence().model_copy(update={"source": "web"})
    result = await AnswerabilityEvaluator(chat_model=model).evaluate("current news", [web])
    assert result.answerable and len(model.prompts) == 1
