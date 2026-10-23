"""本机真实 API 验收：只用 Python 标准库，不读取模型密钥。

python scripts/acceptance.py --seed --output backend/data/evaluation/acceptance-latest.json
密码从 RAGNOTEBOOK_PASSWORD 环境变量或交互输入读取。会创建演示资料和验收会话。
"""
import argparse
import getpass
import json
import os
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--output", default="backend/data/evaluation/acceptance-latest.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = args.base_url.rstrip("/")
    token = ""

    def request(path, payload=None, *, raw=None, content_type=None):
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if body is not None:
            headers["Content-Type"] = content_type or "application/json"
        return urlopen(Request(base + path, data=body, headers=headers), timeout=240)

    def api(path, payload=None):
        with request(path, payload) as response:
            return json.load(response)

    password = os.getenv("RAGNOTEBOOK_PASSWORD") or getpass.getpass("本机账号密码: ")
    login = api("/user/login/", {"username": args.username, "password": password})
    token = login["token"]

    if args.seed:
        documents = api("/knowledge/list")["data"]["documents"]
        existing = {doc["filename"] for doc in documents}
        for path in sorted((root / "demo/documents").glob("*.md")):
            if path.name in existing:
                continue
            boundary = "ragnotebook" + uuid.uuid4().hex
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
                    'Content-Type: text/markdown\r\n\r\n').encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            with request("/knowledge/add/multiple/stream", raw=body, content_type=f"multipart/form-data; boundary={boundary}") as response:
                frames = [json.loads(line[6:]) for line in response if line.startswith(b"data: ")]
            if not any(frame.get("event_type") == "completed" for frame in frames):
                raise RuntimeError(f"上传失败: {path.name}")
            print(f"已登记: {path.name}", flush=True)
        for _ in range(60):
            documents = api("/knowledge/list")["data"]["documents"]
            demo_docs = [doc for doc in documents if doc["filename"] in {p.name for p in (root / "demo/documents").glob("*.md")}]
            if len(demo_docs) == 2 and all(doc["status"] == "completed" and doc["chunk_count"] > 0 for doc in demo_docs):
                break
            if any(doc["status"] == "failed" for doc in demo_docs):
                raise RuntimeError("演示资料处理失败，请在知识库页面检查并重试")
            time.sleep(10)
        else:
            raise TimeoutError("文档尚未处理完成，可稍后重新验收")

    results = []
    for case in json.loads((root / "demo/questions.json").read_text(encoding="utf-8")):
        started = time.perf_counter()
        session_id = str(uuid.uuid4())
        answer, sources, errors = "", [], []
        done = False
        with request("/chat/agent/query/stream", {"query": case["query"], "session_id": session_id}) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("type") == "response":
                    answer += event.get("content", "")
                elif event.get("type") == "sources":
                    sources = event.get("sources", [])
                elif event.get("type") == "error":
                    errors.append(event.get("content", "error"))
                elif event.get("type") == "done":
                    done = True
        elapsed = round(time.perf_counter() - started, 2)
        history = api(f"/chat/session/{session_id}")["data"]
        restored = bool(history["history"] and history["history"][-1][1] == answer
                        and history.get("sources", [[]])[-1] == sources)
        missing = [word for word in case["keywords"] if word.lower() not in answer.lower()]
        passed = done and bool(answer) and not errors and not missing and restored and bool(sources)
        results.append({"id": case["id"], "query": case["query"], "answer": answer,
                        "session_id": session_id, "sources": sources, "seconds": elapsed,
                        "errors": errors, "missing_keywords": missing, "history_restored": restored,
                        "passed": passed})
        print(f"{case['id']}: {'PASS' if passed else 'REVIEW'} ({elapsed}s)", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"scope": "小型演示集关键词与流程检查，不是准确率基准；答案语义需人工复核",
                                  "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
