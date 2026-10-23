"""有界 ANN 扩大、RRF 去重计分与小范围精排的数据库边界测试。"""
from unittest.mock import AsyncMock

import pytest

from app.graph.storage.neo4j_graph_store import Neo4jGraphStore, settings


def chunk(cid):
    return {"c": {"id": cid, "kind": "doc", "source_id": "source", "text": cid}}


async def test_expanding_ann_uses_final_ranking_once_and_merges_fulltext(monkeypatch):
    monkeypatch.setattr(settings, "VECTOR_EXACT_SCOPE_LIMIT", 1)
    monkeypatch.setattr(settings, "VECTOR_ANN_MAX_CANDIDATES", 40)
    store = Neo4jGraphStore()
    run = AsyncMock(side_effect=[
        [{"count": 2}],
        [{"id": "a", "score": 0.9}],
        [{"id": "b", "score": 0.95}, {"id": "a", "score": 0.9}],
        [{"id": "a", "score": 2.0}],
        [chunk("a"), chunk("b")],
    ])
    monkeypatch.setattr(store, "_run", run)
    hits = await store.search_chunks("owner", [1.0, 0.0], "关键词", ["doc"], 1)
    assert [hit.id for hit in hits] == ["a"]
    # a 在最终向量列表排第2，全文排第1，不应因第一次扩大命中重复加分。
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[0].metadata["vector_score"] == 0.9
    assert hits[0].metadata["fulltext_score"] == 2.0
    assert hits[0].metadata["vector_candidate_k"] == 8
    assert hits[0].metadata["vector_pool_filled"]
    assert [call.args[1]["candidate_k"] for call in run.await_args_list if "queryNodes('chunk_embedding_index'" in call.args[0]] == [2, 8]
    assert all(call.args[1]["uid"] == "owner" for call in run.await_args_list)


async def test_ann_stops_at_candidate_cap_and_reports_unfilled_pool(monkeypatch):
    monkeypatch.setattr(settings, "VECTOR_EXACT_SCOPE_LIMIT", 1)
    monkeypatch.setattr(settings, "VECTOR_ANN_MAX_CANDIDATES", 40)
    store = Neo4jGraphStore()
    run = AsyncMock(side_effect=[[{"count": 2}], [], [], [], [{"id": "a", "score": 0.7}], [chunk("a")]])
    monkeypatch.setattr(store, "_run", run)
    hits = await store.search_chunks("owner", [1.0, 0.0], None, ["doc"], 1)
    assert len(hits) == 1 and not hits[0].metadata["vector_pool_filled"]
    assert hits[0].score == pytest.approx(1 / 61)
    assert [call.args[1]["candidate_k"] for call in run.await_args_list if "queryNodes('chunk_embedding_index'" in call.args[0]] == [2, 8, 32, 40]


async def test_empty_scope_skips_global_ann_and_can_still_search_fulltext(monkeypatch):
    store = Neo4jGraphStore()
    run = AsyncMock(side_effect=[[{"count": 0}], [{"id": "a", "score": 1.0}], [chunk("a")]])
    monkeypatch.setattr(store, "_run", run)
    hits = await store.search_chunks("owner", [1.0, 0.0], "关键词", None, 2)
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].metadata == {"fulltext_score": 1.0}
    assert not any("chunk_embedding_index" in call.args[0] for call in run.await_args_list)


@pytest.mark.parametrize("embedding", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]])
async def test_invalid_query_vector_preserves_fulltext_path(embedding, monkeypatch):
    store = Neo4jGraphStore()
    run = AsyncMock(side_effect=[[{"id": "a", "score": 1.0}], [chunk("a")]])
    monkeypatch.setattr(store, "_run", run)
    hits = await store.search_chunks("owner", embedding, "关键词", None, 1)
    assert [hit.id for hit in hits] == ["a"]
    assert not any("embedding" in call.args[1] for call in run.await_args_list)


async def test_small_scope_exact_search_keeps_original_vector_score_and_rrf(monkeypatch):
    store = Neo4jGraphStore()
    run = AsyncMock(side_effect=[[{"count": 2}], [{"id": "a", "score": 0.8}, {"id": "b", "score": 0.7}], [chunk("a"), chunk("b")]])
    monkeypatch.setattr(store, "_run", run)
    hits = await store.search_chunks("owner", [1.0, 0.0], None, ["note", "doc"], 2)
    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].metadata == {"vector_score": 0.8, "vector_search_mode": "scoped_exact", "vector_scope_count": 2}
    assert hits[0].score == pytest.approx(1 / 61)
    assert hits[1].score == pytest.approx(1 / 62)
    assert all(call.args[1]["uid"] == "owner" for call in run.await_args_list)
    assert all(call.args[1]["kinds"] == ["note", "doc"] for call in run.await_args_list[:2])
    assert not any("chunk_embedding_index" in call.args[0] for call in run.await_args_list)
