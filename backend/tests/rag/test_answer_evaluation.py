import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

from app.rag.answer_evaluation import AnswerDataset, mechanical_checks
from scripts import eval_answers

DATASET = Path(__file__).resolve().parents[2] / "evaluation/zhixue_holdout_v1.json"


@pytest.mark.parametrize("answer,terms,matched", [
    ("iloc按位置切片", [["loc"], ["iloc"]], [False, True]),
    ("用loc按标签选择，use LOC here", [["loc"]], [True]),
    ("recount不是count方法", [["count"]], [True]),
    ("recount与nunique", [["count"]], [False]),
    ("默认500条，TTL160分钟", [["50"], ["60"]], [False, False]),
    ("默认50条，TTL 60 分钟", [["50"], ["60"], ["分钟"]], [True, True, True]),
])
def test_mechanical_coverage_respects_identifier_and_numeric_boundaries(answer, terms, matched):
    from app.rag.evaluation import answer_checks

    checks = answer_checks(answer, SimpleNamespace(required_terms=terms))
    assert checks["matched_groups"] == matched


def test_frozen_annotations_cover_sources_and_are_verbatim():
    dataset = AnswerDataset.model_validate_json(DATASET.read_bytes())
    assert len(dataset.documents) == 10 and len(dataset.questions) == 20
    assert sum(q.category == "unanswerable" for q in dataset.questions) == 4
    assert sum(q.category == "multi_source" for q in dataset.questions) == 4


@pytest.mark.parametrize("fault", ["quote", "missing_source", "irrelevant_source", "missing_label"])
def test_rejects_wrong_or_incomplete_ground_truth(fault):
    data = json.loads(DATASET.read_bytes())
    q = data["questions"][0]
    if fault == "quote":
        q["supporting_spans"][0]["quote"] = "a plausible but invented quotation"
    elif fault == "missing_source":
        q["supporting_spans"] = []
    elif fault == "irrelevant_source":
        q["supporting_spans"][0]["source_id"] = "h02"
    else:
        q["relevant_source_ids"] = ["nonexistent"]
    with pytest.raises(ValidationError):
        AnswerDataset.model_validate(data)


def test_valid_number_and_keyword_are_insufficient_to_prove_claim_support():
    q = AnswerDataset.model_validate_json(DATASET.read_bytes()).questions[0]
    # A source ID can be correct while the sentence surrounding its citation is false.
    checks = mechanical_checks("陶知夏的电话是 12345678901。[4]", [{"number": 4, "source_id": "h01"}], q)
    assert checks["keypoint_coverage_heuristic"] == 1
    assert checks["citation_numbers_valid"] and checks["all_required_sources_cited"]
    assert checks["semantic_support"] == "requires_review"
    assert "accuracy" not in checks


@pytest.mark.parametrize("answer,sources", [
    ("陶知夏[0]", [{"number": 1, "source_id": "h01"}]),
    ("陶知夏[2]", [{"number": 1, "source_id": "h01"}]),
    ("陶知夏[1]", [{"number": 1, "source_id": "h01"}, {"number": 1, "source_id": "h02"}]),
])
def test_invalid_or_ambiguous_citation_numbers_fail(answer, sources):
    q = AnswerDataset.model_validate_json(DATASET.read_bytes()).questions[0]
    assert not mechanical_checks(answer, sources, q)["citation_numbers_valid"]


def test_multisource_citations_use_numbers_not_array_positions():
    q = AnswerDataset.model_validate_json(DATASET.read_bytes()).questions[4]
    sources = [{"number": 7, "source_id": "h03"}, {"number": 2, "source_id": "h01"}]
    checks = mechanical_checks("说明[2, 7]", sources, q)
    assert checks["all_required_sources_cited"] and checks["citation_numbers_valid"]
    assert not mechanical_checks("只引一份[2]", sources, q)["all_required_sources_cited"]


async def test_stream_requires_complete_frame_and_does_not_record_error_details():
    async def lines():
        for line in [": ping", "data: " + json.dumps({"type": "response", "content": "回答"}), "",
                     'data: {"type":"error","content":"sensitive-provider-error"}', "",
                     'data: {"type":"done"}']:
            yield line
    row = await eval_answers.consume_stream(lines())
    assert row["answer"] == "回答" and row["error_count"] == 1 and not row["done"]
    assert "sensitive-provider-error" not in json.dumps(row)


async def test_missing_done_is_incomplete_even_when_history_matches():
    q = AnswerDataset.model_validate_json(DATASET.read_bytes()).questions[0]
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, text='data: {"type":"response","content":"陶知夏"}\n\n')
        return httpx.Response(200, json={"data": {"history": [[q.query, "陶知夏"]], "sources": [[]]}})
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        row = await eval_answers.ask(client, q, "test")
    assert row["history_restored"] and row["status"] == "incomplete"


async def test_session_cleanup_rejects_real_or_broad_namespaces():
    for namespace in ("admin", "", "eval-zhixue-", "eval-zhixue-" + "x" * 32):
        with pytest.raises(ValueError):
            await eval_answers.cleanup_sessions(namespace)


async def test_failed_setup_still_cleans_generated_namespace_and_retains_error(monkeypatch, tmp_path):
    store = SimpleNamespace(_run=AsyncMock(return_value=[{"n": 0}]), upsert_chunks=AsyncMock())
    cleanup = AsyncMock()
    monkeypatch.setattr(eval_answers, "Neo4jGraphStore", lambda: store)
    monkeypatch.setattr(eval_answers, "original_fingerprint", AsyncMock(return_value="unchanged"))
    monkeypatch.setattr(eval_answers, "EmbedModelFactory", lambda: SimpleNamespace(generator=lambda: None))
    monkeypatch.setattr(eval_answers, "cleanup_namespace", cleanup)
    monkeypatch.setattr(eval_answers, "cleanup_sessions", AsyncMock(return_value=True))
    monkeypatch.setattr(eval_answers, "close_neo4j_driver", AsyncMock())
    monkeypatch.setattr(eval_answers, "async_engine", SimpleNamespace(dispose=AsyncMock()))
    report = await eval_answers.evaluate(SimpleNamespace(dataset=DATASET, limit=0, interval=7, timeout=180), tmp_path)
    assert report["status"] == "error" and report["error_type"] == "RuntimeError"
    assert all(report["cleanup"].values()) and report["rows"] == []
    assert (tmp_path / "dataset.json").read_bytes() == DATASET.read_bytes()
    cleanup.assert_awaited_once_with(store, report["namespace"])


def test_errors_stay_in_denominators_and_unavailable_is_not_correct_refusal():
    rows = [{"status": "error", "expected_answerable": True, "elapsed_ms": 10},
            {"status": "completed", "expected_answerable": False, "decision": "verification_unavailable", "elapsed_ms": 11}]
    result = eval_answers.summarize(copy.deepcopy(rows))
    assert result["question_count"] == 2 and result["answerable_count"] == 1
    assert result["answerable_generated"] == 0 and result["unanswerable_refused_by_gate"] == 0
