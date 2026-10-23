import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.graph.schemas.graph import ChunkHit
from app.rag.evaluation import FixedDataset, aggregate_rows, answer_checks, retrieval_metrics
from scripts import eval_fixed

DATASET = Path(__file__).resolve().parents[2] / "evaluation" / "zhixue_v1.json"


def test_document_metrics_match_hand_calculation_and_deduplicate_chunks():
    metrics = retrieval_metrics(["wrong", "a", "a", "b"], ["a", "b"])
    assert metrics["recall@1"] == 0 and metrics["recall@3"] == 1
    assert metrics["hit@3"] == 1 and metrics["all_sources_hit@1"] == 0
    assert metrics["all_sources_hit@3"] == 1 and metrics["mrr@8"] == .5
    assert metrics["ndcg@5"] == pytest.approx((1 / math.log2(3) + .5) / (1 + 1 / math.log2(3)))
    assert retrieval_metrics(["a"], []) is None
    assert retrieval_metrics([], ["a"])["mrr@8"] == 0


def test_aggregation_excludes_unanswerable_but_counts_retrieval_failures():
    rows = [{"status": status, "category": category, "baseline": {"metrics": retrieval_metrics(ids, relevant)}}
            for status, category, ids, relevant in [("completed", "fact", ["a"], ["a"]),
                                                    ("error", "fact", [], ["a"]),
                                                    ("completed", "unanswerable", ["a"], [])]]
    result = aggregate_rows(rows, "baseline")
    assert result["questions"] == 3 and result["ir_questions"] == 2
    assert result["metrics"]["recall@5"] == .5 and result["retrieval_failures"] == 1


def test_fixed_dataset_has_expected_coverage():
    dataset = FixedDataset.model_validate_json(DATASET.read_bytes())
    assert len(dataset.documents) == 24 and len(dataset.questions) == 60
    assert {kind: sum(q.category == kind for q in dataset.questions)
            for kind in ("fact", "semantic", "multi_source", "unanswerable")} == {
                "fact": 24, "semantic": 24, "multi_source": 6, "unanswerable": 6}
    assert answer_checks("未提供邮箱地址，无法确定。", dataset.questions[-6])["refusal_phrase_heuristic"]
    assert answer_checks("沈 若 溪。", dataset.questions[0])["keypoint_coverage_heuristic"] == 1


@pytest.mark.parametrize("change", ["duplicate_doc", "duplicate_question", "missing_doc", "empty_relevant", "missing_multisource", "bad_terms"])
def test_dataset_rejects_invalid_annotations(change):
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    if change == "duplicate_doc":
        data["documents"].append(copy.deepcopy(data["documents"][0]))
    elif change == "duplicate_question":
        data["questions"].append(copy.deepcopy(data["questions"][0]))
    elif change == "missing_doc":
        data["questions"][0]["relevant_source_ids"] = ["absent"]
    elif change == "empty_relevant":
        data["questions"][0]["relevant_source_ids"] = []
    elif change == "missing_multisource":
        data["questions"][48]["relevant_source_ids"] = ["d01"]
    else:
        data["questions"][0]["required_terms"] = [[]]
    with pytest.raises(ValidationError):
        FixedDataset.model_validate(data)


@pytest.mark.parametrize("namespace", ["u1", "eval-zhixue-", "eval-zhixue-" + "a" * 32 + "-other"])
async def test_cleanup_rejects_non_run_namespaces(namespace):
    store = SimpleNamespace(_run=AsyncMock())
    with pytest.raises(ValueError):
        await eval_fixed.cleanup_namespace(store, namespace)
    store._run.assert_not_called()


@pytest.mark.parametrize("fail_embedding", [False, True])
async def test_cli_skips_unconfigured_comparison_and_cleans_only_generated_namespace(monkeypatch, fail_embedding):
    store = SimpleNamespace(_run=AsyncMock(), upsert_chunks=AsyncMock(), search_chunks=AsyncMock(return_value=[
        ChunkHit(id="doc:d01:0", kind="doc", source_id="d01", text="沈若溪", score=.03)]))

    class Embeddings:
        def embed_documents(self, texts):
            if fail_embedding:
                raise RuntimeError("provider-error")
            return [[1., 0.] for text in texts]

        def embed_query(self, query):
            return [1., 0.]

    monkeypatch.setattr(eval_fixed, "Neo4jGraphStore", lambda: store)
    monkeypatch.setattr(eval_fixed, "EmbedModelFactory", lambda: SimpleNamespace(generator=lambda: Embeddings()))
    monkeypatch.setattr(eval_fixed, "ensure_graph_schema", AsyncMock())
    monkeypatch.setattr(eval_fixed, "close_neo4j_driver", AsyncMock())
    monkeypatch.setattr(eval_fixed.settings, "RERANKER_API_KEY", "")
    monkeypatch.setattr(eval_fixed.settings, "RERANKER_ENABLED", True)
    report = await eval_fixed.evaluate(SimpleNamespace(dataset=DATASET, limit=2, with_answers=False, candidate_k=20, concurrency=2))
    assert report["status"] == ("error" if fail_embedding else "completed")
    assert report["namespace_cleaned"] is True
    assert report["rerank_comparison"] == {"status": "skipped", "reason": "not_configured"}
    assert "reranked" not in report and "metric_delta" not in report
    cleanup_query, cleanup_params = store._run.call_args.args
    assert cleanup_query == "MATCH (n {user_id: $uid}) DETACH DELETE n"
    assert cleanup_params == {"uid": report["namespace"]}
    assert report["namespace"].startswith("eval-zhixue-")
    assert "not_configured" in eval_fixed.markdown_report(report)
