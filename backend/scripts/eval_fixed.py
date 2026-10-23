"""固定中文题集组件评测：真实向量+全文 RRF，同候选池比较重排，可选回答。"""
import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.core.settings import settings  # noqa: E402
from app.graph.storage.neo4j_client import close_neo4j_driver, ensure_graph_schema  # noqa: E402
from app.graph.storage.neo4j_graph_store import Neo4jGraphStore  # noqa: E402
from app.rag.agentic_rag.evidence import format_evidence_context  # noqa: E402
from app.rag.agentic_rag.local_retriever import LocalRetriever  # noqa: E402
from app.rag.agentic_rag.reranker import CONTEXT_LIMIT, EvidenceReranker  # noqa: E402
from app.rag.evaluation import FixedDataset, aggregate_rows, answer_checks, retrieval_metrics  # noqa: E402
from app.utils.factory import ChatModelFactory, EmbedModelFactory, resolve_chat_config, resolve_embed_config  # noqa: E402


async def cleanup_namespace(store, namespace):
    if not re.fullmatch(r"eval-zhixue-[0-9a-f]{32}", namespace):
        raise ValueError("拒绝清理非本次评测命名空间")
    await store._run("MATCH (n {user_id: $uid}) DETACH DELETE n", {"uid": namespace})


def selected_result(evidences, question):
    # 先截取实际送入上下文的八条 chunk，再去重为文档排名。
    ids = list(dict.fromkeys(item.metadata["source_id"] for item in evidences))
    return {"document_ids": ids, "chunk_ids": [item.id for item in evidences],
            "metrics": retrieval_metrics(ids, question.relevant_source_ids)}


async def generate_answer(chat, question, evidences):
    started = time.perf_counter()
    try:
        answer = await asyncio.wait_for(chat.ainvoke([
            ("system", "你是资料问答助手。只根据给定资料回答，引用资料编号如[1]。问题询问现行规范时不能用归档草案代替。"
             "资料未记载就明确说明未提供或证据不足，禁止猜测或用常识补全。回答简洁。\n资料：\n" + format_evidence_context(evidences)),
            ("human", question.query),
        ]), timeout=90)
        if not isinstance(answer.content, str) or not answer.content.strip():
            raise ValueError("empty_answer")
        return {"status": "completed", "text": answer.content, "checks": answer_checks(answer.content, question),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    except Exception as exc:
        # 异常原文可能包含供应商 URL，不落盘密钥或请求头。
        return {"status": "error", "error_type": type(exc).__name__,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}


async def evaluate(args):
    raw = args.dataset.read_bytes()
    dataset = FixedDataset.model_validate_json(raw)
    questions = dataset.questions[:args.limit] if args.limit else dataset.questions
    namespace = "eval-zhixue-" + uuid.uuid4().hex
    store = Neo4jGraphStore()
    reranker = EvidenceReranker()
    rerank_state = "configured" if reranker.active else "disabled" if not reranker.service.enabled else "not_configured"
    report = {
        "created_at": datetime.now(UTC).isoformat(), "dataset_version": dataset.version,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(), "dataset_description": dataset.description,
        "annotation": dataset.annotation, "document_count": len(dataset.documents), "question_count": len(questions),
        "scope": "组件评测：每篇短文一个 chunk；不含文件解析、planner、图扩展、web、SSE 与会话；耗时非完整问答延迟。",
        "config": {"candidate_k": args.candidate_k, "context_k": CONTEXT_LIMIT, "concurrency": args.concurrency,
                   "embedding_model": resolve_embed_config()["model"],
                   "answer_model": resolve_chat_config()["model"] if args.with_answers else None,
                   "reranker_provider": reranker.service.provider, "reranker_model": reranker.service.model,
                   "reranker_timeout_seconds": reranker.service.timeout,
                   "with_answers": args.with_answers, "web_search": False},
        "rerank_comparison": {"status": "pending" if reranker.active else "skipped", "reason": rerank_state},
        "namespace": namespace, "namespace_cleaned": False, "rows": [],
    }
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        embed = EmbedModelFactory().generator()
        if embed is None:
            raise RuntimeError("embedding_not_configured")
        chat = ChatModelFactory().generator().bind(temperature=0, max_tokens=600) if args.with_answers else None
        await ensure_graph_schema(embed)
        texts = [doc.text for doc in dataset.documents]
        vectors = await asyncio.to_thread(embed.embed_documents, texts)
        if len(vectors) != len(texts) or not all(vectors):
            raise ValueError("invalid_document_embeddings")
        for doc, vector in zip(dataset.documents, vectors):
            await store.upsert_chunks(namespace, "doc", doc.id, doc.title,
                                      [{"chunk_index": 0, "text": doc.text, "embedding": vector}])
        await store._run("CALL db.awaitIndexes(60)")
        await store._run("CALL db.index.fulltext.awaitEventuallyConsistentIndexRefresh()")

        async def run_question(question):
            async with semaphore:
                row = {"id": question.id, "category": question.category, "query": question.query,
                       "relevant_source_ids": question.relevant_source_ids, "reference_answer": question.reference_answer,
                       "required_terms": question.required_terms}
                started = time.perf_counter()
                try:
                    vector = await asyncio.to_thread(embed.embed_query, question.query)
                    if not vector:
                        raise ValueError("empty_query_embedding")
                    hits = await store.search_chunks(namespace, vector, question.query, ["doc"], args.candidate_k)
                    row["retrieval_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    candidates = [LocalRetriever._chunk_to_evidence(hit) for hit in hits]
                    row["candidate_document_ids"] = [hit.source_id for hit in hits]
                    baseline = candidates[:CONTEXT_LIMIT]
                    row["baseline"] = selected_result(baseline, question)
                    # 该基准固定比较 K=8，不能随生产默认值改变对照组条件。
                    ranked = await reranker.rank(question.query, candidates, limit=CONTEXT_LIMIT)
                    row.update({"status": "completed", "rerank_status": ranked.status, "rerank_ms": round(ranked.elapsed_ms, 2)})
                    if reranker.active:
                        row["reranked"] = selected_result(ranked.evidences, question)
                    if chat:
                        row["baseline"]["answer"] = await generate_answer(chat, question, baseline)
                        if reranker.active and ranked.status == "applied":
                            row["reranked"]["answer"] = await generate_answer(chat, question, ranked.evidences)
                except Exception as exc:
                    row.update({"status": "error", "error_type": type(exc).__name__, "rerank_status": "not_run"})
                    row["baseline"] = selected_result([], question)
                    if reranker.active:
                        row["reranked"] = selected_result([], question)
                print(f"{question.id}: {row['status']} / {row['rerank_status']}", flush=True)
                return row

        report["rows"] = await asyncio.gather(*(run_question(question) for question in questions))
        report["baseline"] = aggregate_rows(report["rows"], "baseline")
        report["groups"] = {category: {"baseline": aggregate_rows([r for r in report["rows"] if r["category"] == category], "baseline")}
                            for category in sorted({question.category for question in questions})}
        if reranker.active:
            report["reranked"] = aggregate_rows(report["rows"], "reranked")
            for category, group in report["groups"].items():
                group["reranked"] = aggregate_rows([r for r in report["rows"] if r["category"] == category], "reranked")
            comparable = all(r["status"] == "completed" and r["rerank_status"] in ("applied", "single_candidate") for r in report["rows"])
            report["rerank_comparison"] = {"status": "completed" if comparable else "partial", "reason": "all_applied" if comparable else "contains_failures_or_fallbacks"}
            if comparable:
                report["metric_delta"] = {key: round(report["reranked"]["metrics"][key] - value, 4)
                                          for key, value in report["baseline"]["metrics"].items()}
        report["status"] = "completed" if all(row["status"] == "completed" for row in report["rows"]) else "partial"
        if any(report[variant].get("answer_failures") for variant in ("baseline", "reranked") if variant in report):
            report["status"] = "partial"
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


def markdown_report(report):
    lines = [f"# 固定题集评测：{report['dataset_version']}", "", report["scope"], "",
             f"状态：{report['status']}；文档：{report['document_count']}；题目：{report['question_count']}；隔离数据清理：{report['namespace_cleaned']}。",
             f"题集 SHA256：`{report['dataset_sha256']}`", "",
             f"重排对照：{report['rerank_comparison']['status']} / {report['rerank_comparison']['reason']}。",
             "未配置时不生成重排提升数据；失败回退会单列状态，partial 不代表完整成功对照。", "",
             "| 变体 | IR题数 | Recall@5 | MRR@8 | nDCG@5 | 检索失败 | 回答完成 | 回答失败 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in ("baseline", "reranked"):
        if variant in report:
            item = report[variant]
            metrics = item["metrics"]
            lines.append(f"| {variant} | {item['ir_questions']} | {metrics.get('recall@5', '—')} | {metrics.get('mrr@8', '—')} | "
                         f"{metrics.get('ndcg@5', '—')} | {item['retrieval_failures']} | {item['answers_completed']} | {item['answer_failures']} |")
    lines.extend(["", "检索失败可答题按零分计；无答案题从 IR 均值排除。回答要点覆盖和拒答措辞只是字符串启发式，不能宣称答案准确率或忠实度。",
                  "分组指标、P50/P95、逐题候选、答案、失败与回退状态见同目录 JSON。固定虚构练习集规模有限，不代表真实业务或公开基准。", ""])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BACKEND / "evaluation" / "zhixue_v1.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--candidate-k", type=int, default=settings.RERANKER_CANDIDATE_K)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 表示完整题集")
    parser.add_argument("--with-answers", action="store_true", help="调用现有回答模型，额外消耗 API 配额")
    args = parser.parse_args()
    if not 8 <= args.candidate_k <= 40 or not 1 <= args.concurrency <= 4 or args.limit < 0:
        parser.error("candidate-k 必须为 8～40，concurrency 为 1～4，limit 非负")
    return args


def main():
    args = parse_args()
    report = asyncio.run(evaluate(args))
    output = args.output_dir or BACKEND / "results" / "fixed" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(f"报告：{output}；状态：{report['status']}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
