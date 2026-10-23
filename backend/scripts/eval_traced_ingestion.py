"""Opt-in, process-local API evaluation with private answer-verification traces.

Uses the real FastAPI routes through ASGITransport, including authentication,
retrieval, the Agent, SSE and history. It does not test Nginx or TCP streaming.
An existing deployment must consume indexing tasks. No extra worker, schema
migration, seed user or public debug endpoint is started. Traces contain document
text; they belong in ignored local data, not application logs.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import httpx

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

from eval_ingestion import evaluate  # noqa: E402


class RecordingVerifier:
    def __init__(self, verifier, output):
        self.verifier, self.output = verifier, output
        self.rows = []

    async def verify(self, query, draft, sources):
        diagnostics = {}
        result = await self.verifier().verify(query, draft, sources, diagnostics=diagnostics)
        self.rows.append({"query": query, "answer": result.text, "diagnostics": diagnostics})
        self.output.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


class RecordingEvaluator:
    def __init__(self, evaluator, output):
        self.evaluator, self.output = evaluator, output
        self.rows = []

    async def evaluate(self, query, evidences):
        diagnostics = {}
        result = await self.evaluator(diagnostics=diagnostics).evaluate(query, evidences)
        self.rows.append({"query": query, "status": result.status, "answerable": result.answerable,
                          "diagnostics": diagnostics})
        self.output.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


async def run(args, output):
    from app.agent import agent as agent_module
    from app.core.background_init import init_manager
    from app.db.db_config import async_engine
    from app.db.redis_config import close_redis, connect_redis
    from app.graph.storage.neo4j_client import close_neo4j_driver
    from app.rag.agentic_rag import service as service_module
    from app.rag.agentic_rag.evaluator import AnswerabilityEvaluator
    from app.rag.answer_verifier import AnswerVerifier
    from app.services.database_session_manager import init_database_session_manager
    from main import app

    # ASGITransport does not run the main app's startup/lifespan hooks. Initialize
    # only request dependencies, leaving production ingestion to its sole worker.
    recorder = RecordingVerifier(AnswerVerifier, output / "verification-trace.json")
    evaluator_recorder = RecordingEvaluator(AnswerabilityEvaluator, output / "answerability-trace.json")
    try:
        await init_database_session_manager()
        await connect_redis()
        await init_manager._init_models()
        await init_manager._init_note_service()
        await init_manager._init_vector_store()
        await init_manager._init_reranker()
        with (patch.object(agent_module, "AnswerVerifier", lambda: recorder),
              patch.object(service_module, "AnswerabilityEvaluator", lambda: evaluator_recorder)):
            report = await evaluate(args, output, transport=httpx.ASGITransport(app=app))
        report["transport"] = "in-process ASGI; no Nginx/TCP claim"
        report["verification_trace_count"] = len(recorder.rows)
        report["answerability_trace_count"] = len(evaluator_recorder.rows)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        await close_redis()
        await close_neo4j_driver()
        await async_engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-wait-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.index_wait_seconds <= 1800:
        parser.error("index-wait-seconds must be between 1 and 1800")
    args.base_url = "http://evaluation.local"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(run(args, args.output_dir))
    print(json.dumps({"status": report["status"], "traces": report["verification_trace_count"],
                      "cleanup": report["cleanup"]}, ensure_ascii=False))
    return 0 if report["status"] == "requires_answer_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
