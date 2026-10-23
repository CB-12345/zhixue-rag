"""Local PDF/DOCX upload -> indexing -> production SSE and source-location checks.

Uses a random temporary principal fitting the SQL 36-character user field. It
never selects an existing user or exports credentials. The manifest records whether
the corpus is self-authored or supplied by the user.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from jose import jwt
from sqlalchemy import func, select

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "scripts"))

from eval_answers import ask, original_fingerprint, save, summarize  # noqa: E402

from app.core.logger_handler import logger  # noqa: E402
from app.core.settings import settings  # noqa: E402
from app.db.db_config import AsyncSessionLocal, async_engine  # noqa: E402
from app.graph.storage.neo4j_client import close_neo4j_driver  # noqa: E402
from app.graph.storage.neo4j_graph_store import Neo4jGraphStore  # noqa: E402
from app.models.chat_history import ChatSession  # noqa: E402
from app.models.graph import GraphBuildTask, GraphDoc, GraphExtractLog  # noqa: E402
from app.rag.agentic_rag.evaluator import is_local_only_query  # noqa: E402
from app.rag.answer_evaluation import AnswerQuestion, SupportSpan  # noqa: E402
from app.rag.document_handler.processor import DocumentProcessor  # noqa: E402
from app.utils.factory import resolve_chat_config, resolve_embed_config  # noqa: E402


def load_fixtures(directory):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    files = {}
    for item in manifest["files"]:
        name = item["name"]
        if name != Path(name).name or "/" in name or "\\" in name:
            raise ValueError("fixture_must_be_a_filename")
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("fixture_hash_mismatch")
        files[name] = {"data": data, "source_id": hashlib.md5(data).hexdigest()}
    if not all(is_local_only_query(question["query"]) for question in manifest["questions"]):
        raise ValueError("fixture_questions_must_require_local_evidence")
    return manifest, files


async def inspect_fixtures(directory, namespace):
    def inspect():
        processor = DocumentProcessor(object())
        result = {}
        for name in directory.iterdir():
            if name.suffix not in {".pdf", ".docx"}:
                continue
            docs = processor.get_file_document_sync(str(name), hashlib.md5(name.read_bytes()).hexdigest(), namespace)
            chunks = processor.split_documents_sync(docs)
            result[name.name] = [{"text": c.page_content, "page": c.metadata.get("page")} for c in chunks]
        return result
    return await asyncio.to_thread(inspect)


async def wait_index(namespace, source_id, timeout_seconds=120):
    for _ in range(timeout_seconds):
        async with AsyncSessionLocal() as db:
            row = (await db.execute(select(GraphBuildTask.index_status, GraphBuildTask.status).where(
                GraphBuildTask.user_id == namespace, GraphBuildTask.source_id == source_id,
                GraphBuildTask.source_type == "doc"))).first()
        if row and row[0] == "completed":
            return {"index": row[0], "graph": row[1]}
        if row and row[0] == "failed":
            raise RuntimeError("fixture_index_failed")
        await asyncio.sleep(1)
    raise TimeoutError("fixture_index_timeout")


async def clean(client, store, namespace, files, sessions):
    if not re.fullmatch(r"eval-ingest-[0-9a-f]{24}", namespace):
        raise ValueError("invalid_evaluation_namespace")
    result = {"delete_http_errors": 0}
    for sid in sessions:
        response = await client.delete("/chat/session/" + sid)
        result["delete_http_errors"] += response.status_code not in (200, 404)
    for name in files:
        response = await client.delete("/knowledge/delete/filename", params={"filename": name})
        result["delete_http_errors"] += response.status_code not in (200, 404)
    # Source deletion invalidates its tasks before removing data. Also remove any
    # orphan entities in this strictly generated namespace after API cleanup.
    await store._run("MATCH (n {user_id:$uid}) DETACH DELETE n", {"uid": namespace})
    result["remaining_graph_nodes"] = (await store._run(
        "MATCH (n {user_id:$uid}) RETURN count(n) AS n", {"uid": namespace}))[0]["n"]
    async with AsyncSessionLocal() as db:
        for model in (GraphDoc, GraphBuildTask, GraphExtractLog, ChatSession):
            result["remaining_" + model.__tablename__] = (await db.execute(
                select(func.count()).select_from(model).where(model.user_id == namespace))).scalar_one()
    return result


async def evaluate(args, output, *, transport=None):
    logger.setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    manifest, files = load_fixtures(args.fixtures_dir)
    namespace = "eval-ingest-" + uuid.uuid4().hex[:24]
    store, sessions = Neo4jGraphStore(), []
    report = {"created_at": datetime.now(UTC).isoformat(), "namespace": namespace, "status": "running", "rows": [],
              "scope": ("Real PDF/DOCX files: parse, split, upload, background index, production SSE, citation pages and history. "
                        "Corpus provenance is recorded in the manifest; not a public benchmark."),
              "manifest": manifest, "index_states": {}, "cleanup": {}}
    report["config"] = {"answer_model": resolve_chat_config()["model"], "embedding_model": resolve_embed_config()["model"],
                        "reranker_model": settings.RERANKER_MODEL, "concurrency": 1, "min_start_interval_seconds": 7}
    save(output / "report.json", report)
    before = await original_fingerprint(store, namespace)
    token = jwt.encode({"user_id": namespace, "exp": datetime.now(UTC) + timedelta(hours=2), "jti": uuid.uuid4().hex},
                       settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=180, transport=transport,
                                headers={"Authorization": "Bearer " + token}) as client:
        try:
            report["parsed_chunks"] = await inspect_fixtures(args.fixtures_dir, namespace)
            for question in manifest["questions"]:
                for name, quotes in question["supporting_quotes"].items():
                    text = "".join("".join(c["text"].split()) for c in report["parsed_chunks"][name])
                    if not all("".join(quote.split()) in text for quote in quotes):
                        raise ValueError("expected_fixture_fact_missing_after_parsing")
            for name, item in files.items():
                response = await client.post("/knowledge/add/single", files={"file": (name, item["data"])})
                response.raise_for_status()
                report["index_states"][name] = await wait_index(namespace, item["source_id"],
                                                               getattr(args, "index_wait_seconds", 120))
                print(f"indexed: {name}", flush=True)
            previous = None
            for question in manifest["questions"]:
                if previous is not None:
                    await asyncio.sleep(max(0, 7 - (time.monotonic() - previous)))
                sid = "ingestion-" + uuid.uuid4().hex
                sessions.append(sid)
                q = AnswerQuestion(id=question["id"], category=question["category"], query=question["query"],
                    relevant_source_ids=[files[name]["source_id"] for name in question["relevant_files"]],
                    reference_answer=question["reference_answer"], required_terms=question["required_terms"],
                    supporting_spans=[SupportSpan(source_id=files[name]["source_id"], quote=quote, point="fixture fact")
                                      for name, quotes in question["supporting_quotes"].items() for quote in quotes])
                previous = time.monotonic()
                row = await ask(client, q, sid)
                cited = set(row.get("checks", {}).get("citation_numbers", []))
                sources = [source for source in row.get("sources", []) if source["number"] in cited]
                row["required_pages_cited"] = all(
                    set(pages) <= {s.get("page") for s in sources if s.get("source_id") == files[name]["source_id"]}
                    for name, pages in question["required_cited_pages"].items())
                row["sources_only_fixture_documents"] = all(
                    s.get("source_id") in {item["source_id"] for item in files.values()} for s in row.get("sources", []))
                report["rows"].append(row)
                save(output / "report.json", report)
                print(f"{q.id}: {row['status']} / {row.get('decision', 'error')} / cited_pages={row['required_pages_cited']}", flush=True)
            report["summary"] = summarize(report["rows"])
            report["status"] = "requires_answer_review"
        except Exception as exc:
            report.update(status="error", error_type=type(exc).__name__)
        finally:
            try:
                report["cleanup"] = await clean(client, store, namespace, files, sessions)
                report["cleanup"]["original_registry_and_chunks_unchanged"] = before == await original_fingerprint(store, namespace)
                if any(report["cleanup"][key] != 0 for key in report["cleanup"] if key != "original_registry_and_chunks_unchanged"):
                    report["status"] = "cleanup_requires_attention"
                if not report["cleanup"]["original_registry_and_chunks_unchanged"]:
                    report["status"] = "cleanup_requires_attention"
            except Exception as exc:
                report.update(status="cleanup_requires_attention", cleanup_error_type=type(exc).__name__)
            save(output / "report.json", report)
            await close_neo4j_driver()
            await async_engine.dispose()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=BACKEND / "data/evaluation/ingestion-fixtures")
    parser.add_argument("--base-url", default="http://frontend")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--index-wait-seconds", type=int, default=120,
                        help="Index polling budget for each file; does not change the live worker timeout.")
    args = parser.parse_args()
    if not 1 <= args.index_wait_seconds <= 1800:
        parser.error("index-wait-seconds must be between 1 and 1800")
    output = args.output_dir or BACKEND / "data/evaluation" / ("ingestion-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(evaluate(args, output))
    print(json.dumps({"output": str(output), "status": report["status"], "summary": report.get("summary"),
                      "cleanup": report["cleanup"]}, ensure_ascii=False))
    return 0 if report["status"] == "requires_answer_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
