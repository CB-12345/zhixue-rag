"""Replay frozen answer/source pairs through the real final-answer verifier.

Regression checks, not retrieval/answer-generation accuracy. No user data or DB
writes. Reports preserve the exact input, output and elapsed time for review.
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.core.settings import settings  # noqa: E402
from app.rag.agentic_rag.planner import _resolve_shared_chat_model  # noqa: E402
from app.rag.answer_verifier import AnswerVerifier, _resolve_review_model  # noqa: E402
from app.utils.factory import resolve_chat_config  # noqa: E402
from app.utils.prompt_loader import load_prompt  # noqa: E402


def save(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


async def evaluate(args, output):
    raw = args.dataset.read_bytes()
    dataset = json.loads(raw)
    (output / "dataset.json").write_bytes(raw)
    report = {"created_at": datetime.now(UTC).isoformat(), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "scope": dataset.get("scope", "Self-authored known-failure regression pairs; not a fresh holdout or full chat test."),
              "mode": "generate_from_frozen_sources_then_verify" if args.generate_drafts else "verify_frozen_drafts",
              "model": resolve_chat_config()["model"], "enabled": settings.ANSWER_VERIFICATION_ENABLED,
              "timeout_seconds": settings.ANSWER_VERIFICATION_TIMEOUT_SECONDS, "rows": [], "status": "running"}
    save(output / "report.json", report)
    if not report["enabled"]:
        raise ValueError("verification_must_be_enabled")
    class RecordingModel:
        def __init__(self):
            self.model = _resolve_review_model()
            self.last_review = None

        async def ainvoke(self, messages):
            self.last_review = None
            response = await self.model.ainvoke(messages)
            self.last_review = getattr(response, "content", None)
            return response

    model = RecordingModel()
    verifier = AnswerVerifier(model)
    generator = _resolve_shared_chat_model() if args.generate_drafts else None
    previous = None
    for case in dataset["cases"]:
        if previous is not None:
            await asyncio.sleep(max(0, 7 - (time.monotonic() - previous)))
        previous = time.monotonic()
        diagnostics = {}
        draft = case["draft"]
        if args.generate_drafts:
            context = "\n\n".join(
                f'[{s["number"]}] 来源：{s.get("title", "资料")}\n{s["content"]}' for s in case["sources"])
            response = await asyncio.wait_for(generator.ainvoke([
                SystemMessage(content=load_prompt("rag_context_prompt").replace("{context}", context)),
                HumanMessage(content=case["query"]),
            ]), timeout=settings.AGENT_RUN_TIMEOUT_SECONDS)
            draft = response.content
            if not isinstance(draft, str):
                raise ValueError("generated_draft_not_text")
        result = await verifier.verify(case["query"], draft, case["sources"], diagnostics=diagnostics)
        row = {"id": case["id"], "status": result.status, "answer": result.text,
               "kept_blocks": result.kept_blocks, "removed_blocks": result.removed_blocks,
               "repaired_citations": result.repaired_citations,
               "elapsed_ms": round((time.monotonic() - previous) * 1000, 2),
               "raw_review": model.last_review,
               "diagnostics": diagnostics,
               "expected_status_matched": result.status in case["expected_statuses"],
               "required_terms_present": all(term in result.text for term in case["required_terms"]),
               "forbidden_terms_absent": all(term not in result.text for term in case["forbidden_terms"])}
        report["rows"].append(row)
        save(output / "report.json", report)
        print(f'{case["id"]}: {result.status}', flush=True)
    report["status"] = "requires_answer_review"
    report["mechanical_checks_passed"] = sum(all(row[k] for k in
        ("expected_status_matched", "required_terms_present", "forbidden_terms_absent")) for row in report["rows"])
    save(output / "report.json", report)
    print(json.dumps({"status": report["status"], "cases": len(report["rows"]),
                      "mechanical_checks_passed": report["mechanical_checks_passed"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BACKEND / "evaluation/answer_verification_v1.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--generate-drafts", action="store_true",
                        help="Generate from frozen source snapshots before review; no retrieval, tools or full HTTP chat.")
    args = parser.parse_args()
    output = args.output_dir or BACKEND / "data/evaluation" / ("verification-" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    asyncio.run(evaluate(args, output))


if __name__ == "__main__":
    main()
