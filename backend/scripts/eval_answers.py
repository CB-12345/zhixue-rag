"""Privileged local SSE answer evaluation in an ephemeral, isolated user namespace.

Uses backend configuration in this process; never exports credentials. Direct fixture
indexing excludes upload/parser/chunking quality. Each run writes to a new directory.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from jose import jwt
from sqlalchemy import delete, select, text

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

from eval_fixed import cleanup_namespace  # noqa: E402

from app.core.logger_handler import logger  # noqa: E402
from app.core.settings import settings  # noqa: E402
from app.db.db_config import AsyncSessionLocal, async_engine  # noqa: E402
from app.graph.storage.neo4j_client import close_neo4j_driver, ensure_graph_schema  # noqa: E402
from app.graph.storage.neo4j_graph_store import Neo4jGraphStore  # noqa: E402
from app.models.chat_history import ChatMessage, ChatSession  # noqa: E402
from app.rag.agentic_rag.evaluator import is_local_only_query  # noqa: E402
from app.rag.agentic_rag.service import INSUFFICIENT_EVIDENCE_MESSAGE, VERIFICATION_UNAVAILABLE_MESSAGE  # noqa: E402
from app.rag.answer_evaluation import AnswerDataset, mechanical_checks  # noqa: E402
from app.rag.answer_verifier import UNAVAILABLE_MESSAGE as ANSWER_REVIEW_UNAVAILABLE  # noqa: E402
from app.rag.answer_verifier import UNSUPPORTED_MESSAGE as ANSWER_REVIEW_UNSUPPORTED  # noqa: E402
from app.rag.evaluation import latency_summary  # noqa: E402
from app.utils.factory import EmbedModelFactory, resolve_chat_config, resolve_embed_config  # noqa: E402


def save(path, report):
    path.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())


async def original_fingerprint(store, namespace):
    """Hash original registry/chunk bodies; keep private contents out of reports."""
    async with AsyncSessionLocal() as db:
        docs = (await db.execute(text("SELECT * FROM graph_docs ORDER BY id, user_id"))).mappings().all()
    chunks = await store._run(
        "MATCH (c:Chunk) WHERE c.user_id <> $uid OR c.user_id IS NULL "
        "RETURN c.user_id AS user_id, c.id AS id, c.text AS text ORDER BY user_id, id", {"uid": namespace})
    raw = json.dumps({"docs": [dict(doc) for doc in docs], "chunks": chunks}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


async def consume_stream(lines):
    row = {"answer": "", "sources": [], "done": False, "error_count": 0, "stages": [], "tool_events": 0}
    data = []
    async for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif not line and data:
            event = json.loads("\n".join(data))
            data = []
            kind = event.get("type")
            if kind == "response":
                row["answer"] += event.get("content", "")
            elif kind == "sources":
                row["sources"] = event.get("sources", [])
            elif kind == "done":
                row["done"] = True
            elif kind == "error":
                row["error_count"] += 1
            elif kind in ("tool_call", "tool_result", "approval_required"):
                row["tool_events"] += 1
            if stage := event.get("stage"):
                if stage == "tool_start":
                    row["tool_events"] += 1
                details = event.get("details") or {}
                # Deliberately omit reasoning text, tool arguments and exception messages.
                safe = {key: details[key] for key in ("status", "source", "answerable", "evidence_count",
                        "web_count", "fused_count", "step_count") if key in details}
                row["stages"].append({"stage": stage, "details": safe})
    return row


async def ask(client, question, session_id):
    tick = time.perf_counter()
    row = {"id": question.id, "query": question.query, "category": question.category,
           "expected_answerable": bool(question.relevant_source_ids), "session_id": session_id,
           "status": "error", "history_restored": False}
    try:
        async with client.stream("POST", "/chat/agent/query/stream",
                                 json={"query": question.query, "session_id": session_id}) as response:
            row["http_status"] = response.status_code
            response.raise_for_status()
            row.update(await consume_stream(response.aiter_lines()))
        row["checks"] = mechanical_checks(row["answer"], row["sources"], question)
        row["decision"] = ("answer_verification_unavailable" if row["answer"] == ANSWER_REVIEW_UNAVAILABLE
                           else "answer_unsupported" if row["answer"] == ANSWER_REVIEW_UNSUPPORTED
                           else "verification_unavailable" if row["answer"] == VERIFICATION_UNAVAILABLE_MESSAGE
                           else "refused" if row["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE else "generated")
        restored_response = await client.get("/chat/session/" + session_id)
        restored_response.raise_for_status()
        restored = restored_response.json()["data"]
        row["history_restored"] = (restored["history"] == [[question.query, row["answer"]]]
                                   and restored["sources"] == [row["sources"]])
        row["status"] = "completed" if row["done"] and not row["error_count"] and row["answer"].strip() and row["history_restored"] else "incomplete"
    except Exception as exc:
        row["error_type"] = type(exc).__name__
    row["elapsed_ms"] = round((time.perf_counter() - tick) * 1000, 2)
    return row


async def cleanup_sessions(namespace):
    # Same strict namespace guard as graph cleanup. Never broad-delete real sessions.
    import re
    if not re.fullmatch(r"eval-zhixue-[0-9a-f]{32}", namespace):
        raise ValueError("invalid_evaluation_namespace")
    async with AsyncSessionLocal() as db:
        ids = select(ChatSession.id).where(ChatSession.user_id == namespace)
        await db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(ids)))
        await db.execute(delete(ChatSession).where(ChatSession.user_id == namespace))
        await db.commit()
        return not (await db.execute(select(ChatSession.id).where(ChatSession.user_id == namespace))).first()


def summarize(rows):
    answerable = [row for row in rows if row["expected_answerable"]]
    missing = [row for row in rows if not row["expected_answerable"]]
    return {
        "question_count": len(rows), "answerable_count": len(answerable), "unanswerable_count": len(missing),
        "transport_completed": sum(row["status"] == "completed" for row in rows),
        "decisions": dict(Counter(row.get("decision", "error") for row in rows)),
        "answerable_generated": sum(row.get("decision") == "generated" for row in answerable),
        "answerable_all_required_sources_cited": sum(row.get("checks", {}).get("all_required_sources_cited") is True for row in answerable),
        "answerable_full_keypoint_coverage_heuristic": sum(row.get("checks", {}).get("keypoint_coverage_heuristic") == 1 for row in answerable),
        "unanswerable_refused_by_gate": sum(row.get("decision") == "refused" for row in missing),
        "invalid_citation_rows": sum(row.get("checks", {}).get("citation_numbers_valid") is False for row in rows),
        "elapsed_ms": latency_summary([row["elapsed_ms"] for row in rows]),
        "semantic_review": "pending; mechanical results are not answer accuracy or citation entailment",
    }


async def evaluate(args, output):
    logger.setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    raw = args.dataset.read_bytes()
    dataset = AnswerDataset.model_validate_json(raw)
    questions = dataset.questions[:args.limit] if args.limit else dataset.questions
    if not all(is_local_only_query(question.query) for question in questions):
        raise ValueError("every_question_must_explicitly_require_local_evidence")
    namespace = "eval-zhixue-" + uuid.uuid4().hex
    store = Neo4jGraphStore()
    report = {"created_at": datetime.now(UTC).isoformat(), "dataset_version": dataset.version,
              "dataset_sha256": hashlib.sha256(raw).hexdigest(), "namespace": namespace,
              "scope": ("Real HTTP/SSE, planner, hybrid retrieval, rerank, evidence gate, production Agent and history. "
                        "Direct fixture chunks exclude upload/parsing/chunking and graph extraction. "
                        "Temporary signed principal has no registered account; not a public benchmark."),
              "config": {"embedding_model": resolve_embed_config()["model"], "answer_model": resolve_chat_config()["model"],
                         "reranker_model": settings.RERANKER_MODEL, "concurrency": 1, "min_start_interval_seconds": args.interval,
                         "query_scope": "local-only, enforced by every question", "http_read_timeout_seconds": args.timeout},
              "status": "running", "cleanup": {}, "rows": []}
    # Store exact input before any model call, without overwriting a prior run.
    (output / "dataset.json").write_bytes(raw)
    save(output / "report.json", report)
    before = None
    try:
        before = await original_fingerprint(store, namespace)
        embed = EmbedModelFactory().generator()
        if embed is None:
            raise RuntimeError("embedding_not_configured")
        await ensure_graph_schema(embed)
        vectors = await asyncio.to_thread(embed.embed_documents, [doc.text for doc in dataset.documents])
        if len(vectors) != len(dataset.documents) or not all(vectors):
            raise ValueError("invalid_document_embeddings")
        for doc, vector in zip(dataset.documents, vectors):
            await store.upsert_chunks(namespace, "doc", doc.id, doc.title,
                                      [{"chunk_index": 0, "text": doc.text, "embedding": vector}])
        await store._run("CALL db.awaitIndexes(60)")
        await store._run("CALL db.index.fulltext.awaitEventuallyConsistentIndexRefresh()")
        token = jwt.encode({"user_id": namespace, "exp": datetime.now(UTC) + timedelta(hours=2),
                            "jti": uuid.uuid4().hex}, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout,
                                    headers={"Authorization": "Bearer " + token}) as client:
            previous_start = None
            for question in questions:
                # Respect the live 10/min IP limit. Never change or clear limiter keys.
                if previous_start is not None:
                    await asyncio.sleep(max(0, args.interval - (time.monotonic() - previous_start)))
                previous_start = time.monotonic()
                row = await ask(client, question, "heldout-" + uuid.uuid4().hex)
                report["rows"].append(row)
                save(output / "report.json", report)
                print(f"{question.id}: {row['status']} / {row.get('decision', 'error')}", flush=True)
        report["summary"] = summarize(report["rows"])
        report["status"] = "requires_semantic_review"
    except Exception as exc:
        report.update({"status": "error", "error_type": type(exc).__name__})
    finally:
        try:
            report["cleanup"]["sessions_deleted"] = await cleanup_sessions(namespace)
        except Exception as exc:
            report["cleanup"]["session_error_type"] = type(exc).__name__
        try:
            await cleanup_namespace(store, namespace)
            remaining = await store._run("MATCH (n {user_id:$uid}) RETURN count(n) AS n", {"uid": namespace})
            report["cleanup"]["namespace_deleted"] = remaining == [{"n": 0}]
        except Exception as exc:
            report["cleanup"]["namespace_error_type"] = type(exc).__name__
        if before is not None:
            try:
                report["cleanup"]["original_registry_and_chunks_unchanged"] = before == await original_fingerprint(store, namespace)
            except Exception as exc:
                report["cleanup"]["fingerprint_error_type"] = type(exc).__name__
        if not all(report["cleanup"].get(key) is True for key in
                   ("sessions_deleted", "namespace_deleted", "original_registry_and_chunks_unchanged")):
            report["status"] = "cleanup_requires_attention"
        save(output / "report.json", report)
        await close_neo4j_driver()
        await async_engine.dispose()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BACKEND / "evaluation/zhixue_holdout_v1.json")
    parser.add_argument("--base-url", default="http://frontend")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--interval", type=float, default=7)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.limit < 0 or not 7 <= args.interval <= 60 or not 10 <= args.timeout <= 300:
        parser.error("limit >= 0; interval 7..60; timeout 10..300")
    output = args.output_dir or BACKEND / "results/answers" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(evaluate(args, output))
    print(json.dumps({"output": str(output), "status": report["status"], "summary": report.get("summary"), "cleanup": report["cleanup"]}, ensure_ascii=False))
    return 0 if report["status"] == "requires_semantic_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
