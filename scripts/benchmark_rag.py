"""Small real SSE benchmark; measures answer text separately from thinking frames.

Creates unique chat sessions, checks persisted answers/sources, then deletes them.
Uses the standard library. Credentials are read interactively or from stdin.
"""
import argparse
import concurrent.futures
import getpass
import json
import math
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def summary(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    # Nearest-rank percentiles; keep n explicit, especially for these tiny samples.
    return {"n": len(values), "min": round(values[0], 2), "p50": round(values[math.ceil(.5 * len(values)) - 1], 2),
            "p95": round(values[math.ceil(.95 * len(values)) - 1], 2), "max": round(values[-1], 2)}


def consume_sse(lines, clock, started):
    """Parse complete SSE frames, including CRLF, comments and multi-line data."""
    answer, sources, errors, frames = "", [], 0, []
    first_event, first_answer, done_ms, context_ms = None, None, None, None
    data = []

    def accept():
        nonlocal answer, sources, errors, first_event, first_answer, done_ms, context_ms
        if not data:
            return
        event = json.loads("\n".join(data))
        elapsed = (clock() - started) * 1000
        if first_event is None:
            first_event = elapsed
        kind, stage = event.get("type"), event.get("stage")
        if kind == "response":
            content = event.get("content", "")
            if not isinstance(content, str):
                raise ValueError("Non-text answer")
            answer += content
            if content.strip() and first_answer is None:
                first_answer = elapsed
        elif kind == "sources":
            sources = event.get("sources", [])
        elif kind == "error":
            errors += 1
        elif kind == "done":
            done_ms = elapsed
        if stage:
            details = event.get("details") or {}
            # No prompts, keys, document bodies or reasoning text in the timing trace.
            safe = {key: details[key] for key in ("status", "placeholder", "source", "elapsed_ms", "step_count",
                    "evidence_count", "answerable", "phase", "candidate_count", "selected_count") if key in details}
            frames.append({"stage": stage, "at_ms": round(elapsed, 2), "details": safe})
            if stage == "context_ready":
                context_ms = elapsed

    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n") if isinstance(raw, bytes) else raw.rstrip("\r\n")
        if not line:
            accept()
            data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    # An uncompleted frame at EOF is deliberately ignored.
    return {"answer": answer, "sources": sources, "error_count": errors, "done": done_ms is not None,
            "first_event_ms": first_event, "first_answer_ms": first_answer, "done_ms": done_ms,
            "context_ready_ms": context_ms, "timeline": frames}


def run(args, token):
    base = args.base_url.rstrip("/")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "label": args.label,
              "scope": "Small real SSE workflow sample; not a capacity or accuracy benchmark. No warmup excluded.",
              "query": args.query, "concurrency": args.concurrency, "rounds": args.rounds,
              "timeout_seconds": args.timeout, "groups": [], "rows": []}
    health, stop = [], threading.Event()

    def request(path, payload=None, method=None):
        body = json.dumps(payload).encode() if payload is not None else None
        return urlopen(Request(base + path, data=body, method=method,
                       headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"}), timeout=args.timeout)

    def monitor():
        while not stop.is_set():
            start = time.perf_counter()
            try:
                with urlopen(base + "/health", timeout=5) as response:
                    ok = response.status == 200
                health.append({"ok": ok, "ms": (time.perf_counter() - start) * 1000})
            except Exception as exc:
                health.append({"ok": False, "ms": (time.perf_counter() - start) * 1000, "error_type": type(exc).__name__})
            stop.wait(5)

    def one(concurrency, round_index, barrier):
        session_id = "benchmark-" + uuid.uuid4().hex
        row = {"session_id": session_id, "concurrency": concurrency, "round": round_index,
               "status": "error", "history_restored": False, "session_deleted": False}
        try:
            barrier.wait(timeout=10)
            started = time.perf_counter()
            with request("/chat/agent/query/stream", {"query": args.query, "session_id": session_id}) as response:
                row["http_status"] = response.status
                row.update(consume_sse(response, time.perf_counter, started))
            row["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
            sources = row.pop("sources")
            row["source_count"] = len(sources)
            row["expected_text_found"] = all(word.casefold() in row["answer"].casefold() for word in args.expect)
            row["expected_source_found"] = not args.source_title or any(s.get("title") == args.source_title for s in sources)
            citations = [int(n) for n in re.findall(r"\[(\d+)\]", row["answer"])]
            row["citations_valid"] = all(1 <= n <= len(sources) for n in citations)
            row["has_citations"] = bool(citations)
            with request("/chat/session/" + session_id) as response:
                restored = json.load(response)["data"]
            row["history_restored"] = restored["history"] == [[args.query, row["answer"]]] and restored["sources"] == [sources]
            ok = row["done"] and not row["error_count"] and bool(row["answer"].strip()) and row["history_restored"]
            row["transport_completed"] = ok
            correct = (row["expected_text_found"] and row["expected_source_found"] and row["citations_valid"]
                       and (row["has_citations"] or not args.source_title))
            row["status"] = "passed" if ok and correct else "review_required"
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            if isinstance(exc, HTTPError):
                row["http_status"] = exc.code
            if "started" in locals():
                row["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        finally:
            try:
                with request("/chat/session/" + session_id, method="DELETE") as response:
                    row["session_deleted"] = response.status == 200
            except Exception as exc:
                row["cleanup_error_type"] = type(exc).__name__
            if not row["session_deleted"]:
                row["status"] = "review_required"
        return row

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for concurrency in args.concurrency:
            rows = []
            started = time.perf_counter()
            for round_index in range(args.rounds):
                barrier = threading.Barrier(concurrency)
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(one, concurrency, round_index, barrier) for _ in range(concurrency)]
                    rows.extend(future.result() for future in futures)
            wall_s = time.perf_counter() - started
            report["rows"].extend(rows)
            report["groups"].append({"concurrency": concurrency, "n": len(rows), "passed": sum(r["status"] == "passed" for r in rows),
                "transport_completed": sum(r.get("transport_completed", False) for r in rows), "wall_seconds": round(wall_s, 2),
                **{key: summary([r.get(key) for r in rows]) for key in ("first_event_ms", "first_answer_ms", "context_ready_ms", "total_ms")}})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
            print(json.dumps({"concurrency": concurrency, "passed": report["groups"][-1]["passed"], "n": len(rows),
                              "first_answer_ms": report["groups"][-1]["first_answer_ms"], "total_ms": report["groups"][-1]["total_ms"]}), flush=True)
            if args.batch_gap and concurrency != args.concurrency[-1]:
                stop.wait(args.batch_gap)
    finally:
        stop.set()
        thread.join(timeout=6)
        report["health"] = {"samples": len(health), "failed": sum(not r["ok"] for r in health), "latency_ms": summary([r["ms"] for r in health])}
        report["status"] = "passed" if report["rows"] and all(r["status"] == "passed" for r in report["rows"]) and not report["health"]["failed"] else "review_required"
        args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:3000")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--query", required=True)
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("--source-title")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--batch-gap", type=float, default=0)
    parser.add_argument("--label", default="local")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(1 <= value <= 5 for value in args.concurrency) or not 1 <= args.rounds <= 10 or args.timeout <= 0 or args.batch_gap < 0:
        parser.error("concurrency must be 1..5, rounds 1..10; timeout > 0, batch-gap >= 0")
    if args.token_stdin:
        token = input().strip()
    else:
        password = getpass.getpass("本机账号密码: ")
        body = json.dumps({"username": args.username, "password": password}).encode()
        with urlopen(Request(args.base_url.rstrip("/") + "/user/login/", data=body,
                     headers={"Content-Type": "application/json"}), timeout=30) as response:
            token = json.load(response)["token"]
    result = run(args, token)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
