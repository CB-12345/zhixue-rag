import json
from types import SimpleNamespace

import httpx
import pytest

from app.rag.answer_verifier import UNAVAILABLE_MESSAGE, UNSUPPORTED_MESSAGE
from scripts import eval_verification
from scripts.eval_answers import ask


@pytest.mark.parametrize("answer,decision", [(UNAVAILABLE_MESSAGE, "answer_verification_unavailable"),
                                            (UNSUPPORTED_MESSAGE, "answer_unsupported")])
async def test_real_sse_evaluation_does_not_count_review_fallback_as_a_generated_answer(answer, decision):
    question = SimpleNamespace(id="test", query="开放日", category="fact", relevant_source_ids=["fixture"], required_terms=[["周四"]])

    def transport(request):
        if request.method == "POST":
            content = "".join("data: " + json.dumps(event, ensure_ascii=False) + "\n\n" for event in [
                {"type": "response", "content": answer}, {"type": "done"}])
            return httpx.Response(200, text=content)
        return httpx.Response(200, json={"data": {"history": [[question.query, answer]], "sources": [[]]}})

    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(transport)) as client:
        result = await ask(client, question, "session")
    assert result["decision"] == decision and result["status"] == "completed"
    assert result["checks"]["keypoint_coverage_heuristic"] == 0


@pytest.mark.parametrize("generate", [False, True])
async def test_local_replay_retains_the_draft_raw_review_and_exact_deletion_stage(tmp_path, monkeypatch, generate):
    source = {"number": 1, "title": "临时夹具", "content": "默认容量48条，TTL为75分钟。"}
    draft = "容量48条、TTL75分钟 [1]。\n\n不需要审核 [1]。"
    decisions = {"blocks": [{"index": 0, "keep": True, "quotes": [{"number": 1, "quote": source["content"]}]},
                            {"index": 1, "keep": False, "quotes": []}]}

    async def generation(messages):
        assert source["content"] in messages[0].content
        return SimpleNamespace(content=draft)

    async def review(messages):
        assert json.loads(messages[1].content)["blocks"][1]["text"] == "不需要审核 [1]。"
        return SimpleNamespace(content=json.dumps(decisions))

    monkeypatch.setattr(eval_verification, "_resolve_shared_chat_model", lambda: SimpleNamespace(ainvoke=generation))
    monkeypatch.setattr(eval_verification, "_resolve_review_model", lambda: SimpleNamespace(ainvoke=review))
    monkeypatch.setattr(eval_verification, "load_prompt", lambda name: "参考资料：{context}")
    dataset = tmp_path / "input.json"
    dataset.write_text(json.dumps({"cases": [{"id": "trace", "query": "默认容量和TTL？",
        "draft": "placeholder" if generate else draft, "sources": [source], "expected_statuses": ["filtered"],
        "required_terms": ["48", "75分钟"], "forbidden_terms": ["不需要审核"]}]}), encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    await eval_verification.evaluate(SimpleNamespace(dataset=dataset, generate_drafts=generate), output)
    report = json.loads((output / "report.json").read_bytes())
    row = report["rows"][0]
    assert report["mechanical_checks_passed"] == 1
    assert row["diagnostics"]["draft"] == draft and "placeholder" not in row["answer"]
    assert row["diagnostics"]["raw_review"] == json.dumps(decisions)
    assert row["diagnostics"]["block_results"][1]["reason"] == "model_rejected"
