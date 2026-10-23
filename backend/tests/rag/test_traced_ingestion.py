import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scripts import eval_traced_ingestion as mod


async def test_trace_keeps_separate_request_diagnostics_and_preserves_published_text(tmp_path):
    async def verify(query, draft, sources, *, diagnostics):
        diagnostics.update(draft=draft, raw_review="private decision", query=query)
        return SimpleNamespace(text="verified " + draft)

    output = tmp_path / "private.json"
    recorder = mod.RecordingVerifier(lambda: SimpleNamespace(verify=verify), output)
    for query in ("first", "second"):
        result = await recorder.verify(query, query + " draft", [])
        assert result.text == "verified " + query + " draft"
    rows = json.loads(output.read_bytes())
    assert [row["diagnostics"]["query"] for row in rows] == ["first", "second"]
    assert rows[0]["diagnostics"]["draft"] == "first draft"


async def test_answerability_trace_has_fresh_diagnostics_and_preserves_decision(tmp_path):
    def evaluator(*, diagnostics):
        async def evaluate(query, evidences):
            assert diagnostics == {}
            diagnostics.update(query=query, raw_review="private " + query)
            return SimpleNamespace(status="verified", answerable=query == "first")
        return SimpleNamespace(evaluate=evaluate)

    output = tmp_path / "private-answerability.json"
    recorder = mod.RecordingEvaluator(evaluator, output)
    assert (await recorder.evaluate("first", [])).answerable is True
    assert (await recorder.evaluate("second", [])).answerable is False
    rows = json.loads(output.read_bytes())
    assert [row["diagnostics"]["raw_review"] for row in rows] == ["private first", "private second"]
    assert [row["answerable"] for row in rows] == [True, False]


@pytest.mark.parametrize("fail_at", ["initialization", "evaluation"])
async def test_instrumentation_is_restored_and_connections_closed_on_failure(monkeypatch, tmp_path, fail_at):
    from app.agent import agent as am
    from app.core.background_init import init_manager
    from app.db import db_config, redis_config
    from app.graph.storage import neo4j_client
    from app.rag.agentic_rag import service as sm
    database_session_manager = importlib.import_module("app.services.database_session_manager")

    original = am.AnswerVerifier
    original_evaluator = sm.AnswerabilityEvaluator
    for name in ("_init_models", "_init_note_service", "_init_vector_store", "_init_reranker"):
        monkeypatch.setattr(init_manager, name, AsyncMock())
    monkeypatch.setattr(database_session_manager, "init_database_session_manager", AsyncMock())
    monkeypatch.setattr(redis_config, "connect_redis", AsyncMock())
    close_redis, close_graph, dispose = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(redis_config, "close_redis", close_redis)
    monkeypatch.setattr(neo4j_client, "close_neo4j_driver", close_graph)
    monkeypatch.setattr(db_config, "async_engine", SimpleNamespace(dispose=dispose))
    if fail_at == "initialization":
        init_manager._init_models.side_effect = RuntimeError("initialization failed")

    async def fail(*args, **kwargs):
        assert am.AnswerVerifier is not original
        assert sm.AnswerabilityEvaluator is not original_evaluator
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(mod, "evaluate", fail)
    with pytest.raises(RuntimeError, match="failed"):
        await mod.run(SimpleNamespace(), tmp_path)
    assert am.AnswerVerifier is original
    assert sm.AnswerabilityEvaluator is original_evaluator
    close_redis.assert_awaited_once()
    close_graph.assert_awaited_once()
    dispose.assert_awaited_once()
