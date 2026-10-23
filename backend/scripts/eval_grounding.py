"""使用固定资料核对真实 planner→检索→重排→证据门控；不调用最终回答生成。"""
import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from eval_fixed import cleanup_namespace  # noqa: E402

from app.core.background_init import init_manager  # noqa: E402
from app.core.logger_handler import logger  # noqa: E402
from app.core.settings import settings  # noqa: E402
from app.graph.storage.neo4j_client import close_neo4j_driver, ensure_graph_schema  # noqa: E402
from app.graph.storage.neo4j_graph_store import Neo4jGraphStore  # noqa: E402
from app.rag.agentic_rag.service import AgenticRagService  # noqa: E402
from app.rag.evaluation import FixedDataset, latency_summary  # noqa: E402
from app.utils.factory import ChatModelFactory, EmbedModelFactory, resolve_chat_config, resolve_embed_config  # noqa: E402


async def evaluate(args):
    logger.setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    raw = args.dataset.read_bytes()
    dataset = FixedDataset.model_validate_json(raw)
    questions = dataset.questions[:args.limit] if args.limit else dataset.questions
    namespace = "eval-zhixue-" + uuid.uuid4().hex
    store = Neo4jGraphStore()
    # 仅限制本次评测进程：固定私有资料缺失题不能由实时网页补证。
    settings.WEB_SEARCH_ENABLED = False
    report = {"created_at": datetime.now(UTC).isoformat(), "dataset_version": dataset.version,
              "dataset_sha256": hashlib.sha256(raw).hexdigest(), "question_count": len(questions),
              "scope": "实际 planner、向量/全文检索、重排和原文依据核对；不含最终回答生成、SSE 与历史持久化。不是答案准确率。",
              "config": {"embedding_model": resolve_embed_config()["model"], "evaluation_model": resolve_chat_config()["model"],
                         "reranker_model": settings.RERANKER_MODEL, "concurrency": args.concurrency, "web_search": False},
              "namespace": namespace, "namespace_cleaned": False, "rows": []}
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        embed = EmbedModelFactory().generator()
        if embed is None:
            raise RuntimeError("embedding_not_configured")
        init_manager.embed_model = embed
        init_manager.chat_model = ChatModelFactory().generator()
        await ensure_graph_schema(embed)
        vectors = await asyncio.to_thread(embed.embed_documents, [doc.text for doc in dataset.documents])
        if len(vectors) != len(dataset.documents) or not all(vectors):
            raise ValueError("invalid_document_embeddings")
        for doc, vector in zip(dataset.documents, vectors):
            await store.upsert_chunks(namespace, "doc", doc.id, doc.title,
                                      [{"chunk_index": 0, "text": doc.text, "embedding": vector}])
        await store._run("CALL db.awaitIndexes(60)")
        await store._run("CALL db.index.fulltext.awaitEventuallyConsistentIndexRefresh()")

        async def run_question(question):
            async with semaphore:
                started = time.perf_counter()
                row = {"id": question.id, "query": question.query, "category": question.category,
                       "expected_answerable": bool(question.relevant_source_ids)}
                try:
                    result = await AgenticRagService().run(question.query, namespace)
                    allowed = bool(result.plan.need_retrieval and result.evidences and not result.refusal_message)
                    decision = "allowed" if allowed else "refused" if result.refusal_message else "retrieval_skipped"
                    by_id = {item.id: item for item in result.evidences}
                    quotes = result.answerability.supporting_quotes if result.answerability else []
                    quotes_verified = bool(quotes) and all(
                        item.evidence_id in by_id and "".join(item.quote.split()).casefold()
                        in "".join(by_id[item.evidence_id].content.split()).casefold() for item in quotes)
                    row.update({"status": "completed", "decision": decision,
                                "verifier_status": result.answerability.status if result.answerability else "not_run",
                                "matches_annotation": (allowed == row["expected_answerable"] and decision != "retrieval_skipped"),
                                "quotes_verified": quotes_verified, "plan": result.plan.model_dump(),
                                "answerability": result.answerability.model_dump() if result.answerability else None,
                                "source_ids": [item.metadata.get("source_id", item.id) for item in result.evidences],
                                "refusal_message": result.refusal_message, "rerank": result.rerank})
                except Exception as exc:
                    row.update({"status": "error", "error_type": type(exc).__name__, "matches_annotation": False})
                row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                print(f"{question.id}: {row['status']} / {row.get('decision', 'error')} / expected={row['expected_answerable']}", flush=True)
                return row

        report["rows"] = await asyncio.gather(*(run_question(question) for question in questions))
        rows = report["rows"]
        report["summary"] = {
            "expected_answerable": sum(row["expected_answerable"] for row in rows),
            "expected_unanswerable": sum(not row["expected_answerable"] for row in rows),
            "matches_annotation": sum(row["matches_annotation"] for row in rows),
            "false_accepts": sum(not row["expected_answerable"] and row.get("decision") == "allowed" for row in rows),
            "false_refusals": sum(row["expected_answerable"] and row.get("decision") == "refused" for row in rows),
            "verifier_status_counts": dict(Counter(row.get("verifier_status", "error") for row in rows)),
            "allowed_without_verified_quotes": sum(row.get("decision") == "allowed" and not row["quotes_verified"] for row in rows),
            "pipeline_ms": latency_summary([row["elapsed_ms"] for row in rows]),
        }
        report["status"] = "completed" if all(row["matches_annotation"] and row.get("verifier_status") != "unavailable"
                                              and (row.get("decision") != "allowed" or row["quotes_verified"]) for row in rows) else "review_required"
    except Exception as exc:
        report.update({"status": "error", "error_type": type(exc).__name__})
    finally:
        try:
            await cleanup_namespace(store, namespace)
            report["namespace_cleaned"] = True
        except Exception as exc:
            report.update({"status": "error", "cleanup_error_type": type(exc).__name__})
        await close_neo4j_driver()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BACKEND / "evaluation/zhixue_v1.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.concurrency <= 4:
        parser.error("limit 必须非负；concurrency 为 1～4")
    output = args.output_dir or BACKEND / "results/grounding" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(evaluate(args))
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告：{output}；状态：{report['status']}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
