import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.rag import answer_verifier as mod

SOURCES = [{"number": 1, "source": "knowledge_base", "title": "课程规则", "content": "REQ-022 在36小时内反馈。"}]


def keep(index=0, quote="REQ-022 在36小时内反馈。", number=1):
    return {"index": index, "keep": True, "quotes": [{"number": number, "quote": quote}]}


def model_for(blocks):
    return SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({"blocks": blocks}))))


@pytest.mark.parametrize("url,should_bind", [("https://api.deepseek.com/v1", True), ("https://api.example.test/v1", False)])
def test_non_thinking_review_is_scoped_to_official_provider_and_does_not_mutate_shared_client(monkeypatch, url, should_bind):
    model = Mock()
    monkeypatch.setattr(mod, "_resolve_shared_chat_model", lambda: model)
    monkeypatch.setattr(mod.settings, "OPENAI_BASE_URL", url)
    result = mod._resolve_review_model()
    if should_bind:
        model.bind.assert_called_once_with(extra_body={"thinking": {"type": "disabled"}})
        assert result is model.bind.return_value
    else:
        model.bind.assert_not_called()
        assert result is model


async def test_correct_fact_is_preserved_and_uncited_prose_gets_verified_number():
    result = await mod.AnswerVerifier(model_for([keep()])).verify("反馈多久？", "36小时内反馈。", SOURCES)
    assert result.text == "36小时内反馈。 [1]" and result.status == "supported"


@pytest.mark.parametrize("marker", ["```", "~~~"])
async def test_verified_code_citation_does_not_keep_following_explanation_inside_code(marker):
    from markdown_it import MarkdownIt

    code = 'df = s.to_frame()  # 列名 hello_python'
    sources = [{"number": 1, "content": code}]
    draft = f"{marker}python\n{code}\n{marker} [1]\n\n列名为 hello_python [1]。"
    result = await mod.AnswerVerifier(model_for([
        keep(0, quote=code), keep(1, quote=code),
    ])).verify("列名是什么？", draft, sources)
    tokens = MarkdownIt().parse(result.text)
    fences = [token for token in tokens if token.type == "fence"]
    assert result.status == "supported" and fences[0].content == code + "\n"
    assert any(token.type == "inline" and "列名为 hello_python" in token.content for token in tokens)
    assert result.text.count("[1]") == 2


async def test_valid_fence_and_literal_shorter_fence_in_code_are_preserved():
    draft = "````text\n``` [1]\n````\n[1]"
    result = await mod.AnswerVerifier(model_for([keep()])).verify("示例", draft, SOURCES)
    assert result.status == "supported" and result.text == draft


async def test_fixed_fact_free_introduction_does_not_create_a_false_omission_warning():
    result = await mod.AnswerVerifier(model_for([keep()])).verify("时限", "根据你的资料：\n\n36小时内反馈[1]", SOURCES)
    assert result.text == "36小时内反馈[1]" and result.status == "supported"


async def test_unsupported_negative_paragraph_is_removed_without_regenerating_supported_core():
    draft = "36小时内反馈。[1]\n\n此外，不需要任何审核。"
    result = await mod.AnswerVerifier(model_for([keep(), {"index": 1, "keep": False, "quotes": []}])).verify("时限", draft, SOURCES)
    assert result.text == "36小时内反馈。[1]\n\n" + mod.OMISSION_NOTICE
    assert result.status == "filtered" and result.removed_blocks == 1


@pytest.mark.parametrize("quote,number", [("无需审核", 1), ("36小时", 99), ("", 1), ("36小时", True)])
async def test_model_cannot_approve_fabricated_quote_or_unknown_source(quote, number):
    result = await mod.AnswerVerifier(model_for([keep(quote=quote, number=number)])).verify("时限", "无需审核[1]", SOURCES)
    assert result.text == mod.UNSUPPORTED_MESSAGE and result.status == "unsupported"


async def test_every_cited_source_must_have_a_valid_quote():
    result = await mod.AnswerVerifier(model_for([keep()])).verify("时限", "36小时内反馈[1]，另有其他规定[2]", SOURCES)
    assert result.status == "unsupported"


@pytest.mark.parametrize("citations", ["[1][99]", "[1,99]", "[1，99]"])
async def test_redundant_citation_can_be_deleted_only_when_its_group_keeps_verified_support(citations):
    result = await mod.AnswerVerifier(model_for([keep()])).verify("时限", "36小时内反馈" + citations, SOURCES)
    assert result.text == "36小时内反馈[1]" and result.repaired_citations == 1


async def test_review_cannot_silently_use_an_uncited_source_to_justify_a_block():
    sources = [*SOURCES, {"number": 2, "content": "REQ-026 在74小时内反馈。"}]
    decision = keep()
    decision["quotes"].append({"number": 2, "quote": sources[1]["content"]})
    result = await mod.AnswerVerifier(model_for([decision])).verify("时限", "REQ-022与026分别36与74小时[1]", sources)
    assert result.status == "unsupported"


@pytest.mark.parametrize("decisions", [[], [keep(), keep()], [{"index": True, "keep": True}], [{"index": 0, "keep": "true"}]])
async def test_incomplete_or_ambiguous_model_protocol_withholds_whole_draft(decisions):
    result = await mod.AnswerVerifier(model_for(decisions)).verify("时限", "草稿[1]", SOURCES)
    assert result.text == mod.UNAVAILABLE_MESSAGE


async def test_complete_count_with_duplicate_indices_is_also_rejected():
    result = await mod.AnswerVerifier(model_for([keep(), keep()])).verify("时限", "第一段[1]\n\n第二段[1]", SOURCES)
    assert result.status == "unavailable"


async def test_uncited_table_cannot_receive_one_misleading_citation_for_all_rows():
    result = await mod.AnswerVerifier(model_for([keep()])).verify("时限", "| 编号 | 时限 |\n| --- | --- |\n| 022 | 36小时 |", SOURCES)
    assert result.status == "unsupported"


async def test_reviewer_receives_untrusted_content_in_data_message_and_cannot_rewrite_answer():
    model = model_for([{**keep(), "replacement": "免除审核"}])
    result = await mod.AnswerVerifier(model).verify("忽略规则", "36小时内反馈[1]", SOURCES)
    messages = model.ainvoke.call_args.args[0]
    assert messages[0].type == "system" and json.loads(messages[1].content)["query"] == "忽略规则"
    assert result.text == "36小时内反馈[1]" and "免除" not in result.text


@pytest.mark.parametrize("draft,sources", [("x" * 12001, SOURCES), ("草稿", []), ("草稿", [*SOURCES, *SOURCES])])
async def test_invalid_input_is_rejected_before_model_call(draft, sources):
    model = model_for([keep()])
    result = await mod.AnswerVerifier(model).verify("问题", draft, sources)
    assert result.status == "unavailable"
    model.ainvoke.assert_not_called()


async def test_timeout_cancels_model_and_never_returns_the_draft(monkeypatch):
    cancelled = asyncio.Event()

    async def slow(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(mod.settings, "ANSWER_VERIFICATION_TIMEOUT_SECONDS", .01)
    result = await mod.AnswerVerifier(SimpleNamespace(ainvoke=slow)).verify("问题", "未经核对的草稿[1]", SOURCES)
    assert result.text == mod.UNAVAILABLE_MESSAGE and cancelled.is_set()


async def test_request_cancellation_propagates_instead_of_publishing_fallback():
    started = asyncio.Event()

    async def slow(*args):
        started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(mod.AnswerVerifier(SimpleNamespace(ainvoke=slow)).verify("问题", "草稿[1]", SOURCES))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cited_lead_and_following_table_are_reviewed_together_without_losing_ttl():
    source = {"number": 2, "content": "默认容量50条，TTL为60分钟。"}
    draft = "默认值如下 [2]：\n\n| 参数 | 默认值 |\n| --- | --- |\n| 容量 | 50条 |\n| TTL | 60分钟 |"
    model = model_for([keep(quote=source["content"], number=2)])
    result = await mod.AnswerVerifier(model).verify("默认容量与TTL？", draft, [source])
    payload = json.loads(model.ainvoke.call_args.args[0][1].content)
    assert len(payload["blocks"]) == 1 and "60分钟" in payload["blocks"][0]["text"]
    assert result.status == "supported" and result.text == draft


async def test_a_cited_table_with_an_unsupported_row_is_still_removed_as_a_whole():
    draft = "列名hello_python [1]。\n\n默认值如下 [2]：\n\n| 参数 | 默认值 |\n| --- | --- |\n| TTL | 600分钟 |"
    sources = [{"number": 1, "content": "列名hello_python。"}, {"number": 2, "content": "TTL为60分钟。"}]
    model = model_for([keep(quote=sources[0]["content"]), {"index": 1, "keep": False, "quotes": []}])
    result = await mod.AnswerVerifier(model).verify("列名与TTL？", draft, sources)
    assert result.status == "filtered" and "hello_python" in result.text and "600" not in result.text


async def test_blank_lines_inside_fenced_code_do_not_split_the_verification_unit():
    draft = "```python\ncapacity = 50\n\nttl = 60\n``` [1]"
    sources = [{"number": 1, "content": "capacity = 50\nttl = 60"}]
    result = await mod.AnswerVerifier(model_for([keep(quote=sources[0]["content"])])).verify("原代码", draft, sources)
    assert result.status == "supported" and "capacity = 50\n\nttl = 60\n```\n[1]" in result.text


async def test_explicitly_cited_lead_can_also_support_its_following_code_block():
    draft = "原始代码 [1]：\n\n```python\ncapacity = 50\n```"
    sources = [{"number": 1, "content": "capacity = 50"}]
    result = await mod.AnswerVerifier(model_for([keep(quote=sources[0]["content"])])).verify("原代码", draft, sources)
    assert result.status == "supported" and result.text == draft


@pytest.mark.parametrize("target,original,changed", [
    ("per", "max(1, pool // max(1, len(expansions)))", "pool // len(expansions)"),
    ("limit", "max(2, n // max(1, m))", "max(2, n // m)"),
    ("self.ttl", "config.ttl or 75", "75"),
])
async def test_changed_assignment_is_blocked_even_when_the_model_approves(target, original, changed):
    code = f"{target} = {original}"
    sources = [{"number": 1, "content": code}]
    diagnostics = {}
    result = await mod.AnswerVerifier(model_for([keep(quote=code)])).verify(
        "原公式？", f"`{target} = {changed}` [1]", sources, diagnostics=diagnostics)
    assert result.status == "unsupported" and changed not in result.text
    assert diagnostics["block_results"] == [{"index": 0, "reason": "changed_code_assignment"}]


async def test_false_approved_code_is_removed_but_a_separately_supported_fact_survives():
    code = "per = max(1, pool // max(1, len(expansions)))"
    sources = [{"number": 1, "content": "默认倍数为4。\n" + code}]
    draft = "默认倍数为4 [1]。\n\n`per = pool // len(expansions)` [1]"
    result = await mod.AnswerVerifier(model_for([
        keep(quote="默认倍数为4。"), keep(1, quote=code),
    ])).verify("默认倍数", draft, sources)
    assert result.status == "filtered" and "倍数为4" in result.text and "pool //" not in result.text


async def test_assignment_matching_ignores_spacing_comments_and_python_quote_style():
    source = {"number": 1, "content": '| Python<br>name = "demo"<br>per = max(1, pool // max(1, size)) |'}
    draft = "```python\nname='demo'\nper=max(1,pool//max(1,size))  # 原始边界保护\n``` [1]"
    result = await mod.AnswerVerifier(model_for([keep(quote=source["content"])])).verify("原代码", draft, [source])
    assert result.status == "supported" and "max(1,size)" in result.text


async def test_source_with_alternative_same_name_in_an_uncited_document_cannot_repair_formula():
    correct = "per = max(1, pool // max(1, size))"
    changed = "per = pool // size"
    sources = [{"number": 1, "content": correct}, {"number": 2, "content": changed}]
    result = await mod.AnswerVerifier(model_for([keep(quote=correct)])).verify("原公式", f"`{changed}` [1]", sources)
    assert result.status == "unsupported"


async def test_code_comparison_never_executes_document_or_draft_code(tmp_path):
    marker = tmp_path / "must-not-exist.txt"
    code = f"result = __import__('pathlib').Path({str(marker)!r}).write_text('executed')"
    result = await mod.AnswerVerifier(model_for([keep(quote=code)])).verify(
        "保留代码但不要执行", f"```python\n{code}\n``` [1]", [{"number": 1, "content": code}])
    assert result.status == "supported" and not marker.exists()


async def test_opt_in_local_diagnostics_distinguish_model_rejection_from_program_rejection():
    diagnostics = {"stale": "discard"}
    decisions = [keep(), {"index": 1, "keep": False, "quotes": []}]
    result = await mod.AnswerVerifier(model_for(decisions)).verify(
        "时限", "36小时内反馈[1]。\n\n无需审核[1]。", SOURCES, diagnostics=diagnostics)
    assert result.status == "filtered" and "stale" not in diagnostics
    assert diagnostics["raw_review"] == json.dumps({"blocks": decisions})
    assert [r["reason"] for r in diagnostics["block_results"]] == ["kept", "model_rejected"]
    assert diagnostics["review_blocks"][1] == "无需审核[1]。"


async def test_a_wrong_independently_cited_list_item_does_not_discard_other_items():
    sources = [{"number": 1, "content": "默认容量48条，TTL为75分钟。"},
               {"number": 2, "content": "列名为study_name。"}]
    draft = "- 容量48条 [1]。\n- TTL750分钟 [1]。\n- 列名study_name [2]。"
    decisions = [keep(quote="默认容量48条"), {"index": 1, "keep": False, "quotes": []},
                 keep(2, quote="列名为study_name。", number=2)]
    diagnostics = {}
    result = await mod.AnswerVerifier(model_for(decisions)).verify("三个参数？", draft, sources, diagnostics=diagnostics)
    assert result.status == "filtered" and result.kept_blocks == 2 and result.removed_blocks == 1
    assert "48条" in result.text and "study_name" in result.text and "750" not in result.text
    assert diagnostics["review_blocks"] == draft.splitlines()


@pytest.mark.parametrize("prefix", ["-", "+", "*", "1.", "1)"])
def test_list_units_preserve_nested_items_and_continuation_lines(prefix):
    draft = f"{prefix} 容量48条 [1]：\n  - 重启后丢失\n  同一条的说明。\n{prefix} TTL75分钟 [1]。"
    blocks = mod._review_blocks(draft)
    assert len(blocks) == 2 and "重启后丢失\n  同一条" in blocks[0]


def test_a_shared_final_citation_does_not_get_copied_to_preceding_items():
    draft = "- 容量48条。\n- 列名study_name [2]。"
    assert mod._review_blocks(draft) == [draft]


def test_fenced_code_inside_a_list_item_is_not_split_at_literal_bullet_lines():
    draft = '- 原始示例 [1]：\n  ```text\n- 代码内的文本 [2]\n\n- 更多文本\n  ```\n- 另一条事实 [3]。'
    blocks = mod._review_blocks(draft)
    assert len(blocks) == 2 and "\n- 代码内的文本 [2]\n\n- 更多文本" in blocks[0]
    assert blocks[0].endswith("  ```")


async def test_reviewer_cannot_borrow_a_siblings_source_for_a_wrong_item():
    draft = "- 容量48条 [2]。\n- 列名study_name [2]。"
    sources = [{"number": 1, "content": "容量48条。"}, {"number": 2, "content": "列名study_name。"}]
    result = await mod.AnswerVerifier(model_for([
        keep(quote="容量48条。"), keep(1, quote="列名study_name。", number=2),
    ])).verify("参数？", draft, sources)
    assert result.status == "filtered" and "容量" not in result.text and "study_name" in result.text


async def test_heading_does_not_make_independently_cited_items_one_unit():
    draft = '**参数说明**\n- 容量48条 [1]。\n- TTL750分钟 [1]。'
    result = await mod.AnswerVerifier(model_for([
        keep(quote="容量48条"), {"index": 1, "keep": False, "quotes": []},
    ])).verify("参数？", draft, [{"number": 1, "content": "容量48条，TTL75分钟。"}])
    assert result.status == "filtered" and "容量48条" in result.text and "750分钟" not in result.text


def test_a_heading_reference_is_not_inherited_by_an_uncited_item():
    draft = '**参数说明 [1]**\n- 容量48条。\n- TTL75分钟 [1]。'
    assert mod._review_blocks(draft) == [draft]


async def test_cited_table_rows_are_reviewed_separately_and_rebuilt_as_one_valid_table():
    from markdown_it import MarkdownIt

    draft = '| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 [1] |\n| TTL | 750分钟 [1] |\n| 列名 | study [2] |'
    sources = [{"number": 1, "content": "容量48条，TTL75分钟。"}, {"number": 2, "content": "列名study。"}]
    diagnostics = {}
    result = await mod.AnswerVerifier(model_for([
        keep(quote="容量48条"), {"index": 1, "keep": False, "quotes": []},
        keep(2, quote="列名study。", number=2),
    ])).verify("参数？", draft, sources, diagnostics=diagnostics)
    assert result.status == "filtered" and result.kept_blocks == 2 and result.removed_blocks == 1
    assert result.text.count('| 参数 | 值 |') == 1 and '750' not in result.text and 'study' in result.text
    tokens = MarkdownIt().enable('table').parse(result.text)
    assert sum(t.type == 'table_open' for t in tokens) == 1
    assert sum(t.type == 'tr_open' for t in tokens) == 3  # header plus two surviving rows
    assert len(diagnostics['review_blocks']) == 3 and len(diagnostics['table_groups']) == 3


async def test_all_supported_table_rows_remain_in_original_order():
    draft = '| 参数 | 值 |\n| :--- | ---: |\n| 容量 | 48条 [1] |\n| TTL | 75分钟 [1] |'
    result = await mod.AnswerVerifier(model_for([
        keep(quote="容量48条"), keep(1, quote="TTL75分钟"),
    ])).verify("参数？", draft, [{"number": 1, "content": "容量48条，TTL75分钟。"}])
    assert result.status == 'supported' and result.text == draft


@pytest.mark.parametrize('draft', [
    '参数 [1]：\n\n| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 |\n| TTL | 75分钟 |',
    '| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 [1] |\n| TTL | 75分钟 |',
    '| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 [1] |\n| TTL | 75分钟 [1] | 额外单元格 |',
    '```markdown\n| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 [1] |\n| TTL | 75分钟 [1] |\n```',
])
def test_ambiguous_shared_or_fenced_tables_keep_the_original_scope(draft):
    blocks, groups = mod._review_units(draft)
    assert len(blocks) == 1 and groups == {}


async def test_factual_lead_is_checked_with_every_table_row_and_cannot_bypass_review():
    draft = '无需审核 [1]：\n| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 [1] |\n| TTL | 75分钟 [1] |'
    model = model_for([{"index": i, "keep": False, "quotes": []} for i in range(2)])
    result = await mod.AnswerVerifier(model).verify("参数？", draft, [{"number": 1, "content": "容量48条，TTL75分钟。"}])
    blocks = json.loads(model.ainvoke.call_args.args[0][1].content)['blocks']
    assert all('无需审核' in b['text'] for b in blocks) and result.status == 'unsupported'


async def test_correct_docx_code_quote_with_plain_newlines_is_not_deleted():
    source = {"number": 1, "content": '# loc[] 标签<br>print(s.loc["c"])<br># iloc[] 位置'}
    quote = '# loc[] 标签\nprint(s.loc["c"])\n# iloc[] 位置'
    result = await mod.AnswerVerifier(model_for([keep(quote=quote)])).verify(
        '索引方式？', 'loc按标签、iloc按位置 [1]。', [source])
    assert result.status == 'supported'


@pytest.mark.parametrize("http_status", [402, "secret-response", True])
async def test_private_failure_trace_only_records_a_numeric_http_status(http_status, monkeypatch):
    class ProviderFailure(Exception):
        status_code = http_status
        body = {"error": "secret-provider-body"}

    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=ProviderFailure("secret-key-in-message")))
    warning = Mock()
    monkeypatch.setattr(mod.logger, "warning", warning)
    diagnostics = {}
    result = await mod.AnswerVerifier(model).verify("时限？", "36小时内反馈[1]", SOURCES, diagnostics=diagnostics)
    assert result.status == "unavailable" and result.text == mod.UNAVAILABLE_MESSAGE
    assert diagnostics.get("http_status") == (402 if type(http_status) is int else None)
    assert "secret-" not in json.dumps(diagnostics) + str(warning.call_args)
    model.ainvoke.assert_awaited_once()


@pytest.mark.parametrize("heading", ["## 默认值", "**默认值**", "__默认值__", "**默认值**："])
async def test_standalone_heading_travels_with_its_reviewed_fact(heading):
    draft = heading + "\n\n36小时内反馈[1]。"
    model = model_for([keep()])
    result = await mod.AnswerVerifier(model).verify("反馈多久？", draft, SOURCES)
    payload = json.loads(model.ainvoke.call_args.args[0][1].content)
    assert payload['blocks'] == [{'index': 0, 'text': draft}]
    assert result.status == "supported" and result.text == draft and result.removed_blocks == 0


async def test_factual_heading_is_reviewed_and_removed_while_supported_body_survives():
    draft = "## 无需审批 [1]\n\n36小时内反馈[1]。"
    model = model_for([{'index': 0, 'keep': False, 'quotes': []}, keep(1)])
    result = await mod.AnswerVerifier(model).verify("反馈多久？", draft, SOURCES)
    payload = json.loads(model.ainvoke.call_args.args[0][1].content)
    assert payload['blocks'][0]['text'] == '## 无需审批 [1]' and len(payload['blocks']) == 2
    assert result.status == "filtered" and '无需审批' not in result.text and '36小时' in result.text


async def test_missing_a_fact_still_invalidates_the_whole_review_after_heading_grouping():
    draft = "## 默认值\n\n36小时内反馈[1]。\n\n无需审批[1]。"
    result = await mod.AnswerVerifier(model_for([keep()])).verify("规定？", draft, SOURCES)
    assert result.status == "unavailable" and result.text == mod.UNAVAILABLE_MESSAGE


async def test_factual_table_heading_keeps_its_own_required_review():
    draft = '## 无需审批 [1]\n\n| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 [1] |\n| TTL | 75分钟 [1] |'
    model = model_for([{'index': 0, 'keep': False, 'quotes': []},
                       keep(1, quote='容量48条'), keep(2, quote='TTL75分钟')])
    result = await mod.AnswerVerifier(model).verify("参数？", draft, [{"number":1,"content":"容量48条，TTL75分钟。"}])
    blocks = json.loads(model.ainvoke.call_args.args[0][1].content)['blocks']
    assert len(blocks) == 3 and blocks[0]['text'] == '## 无需审批 [1]'
    assert result.status == 'filtered' and '无需审批' not in result.text and '48条' in result.text and '75分钟' in result.text


def test_heading_reference_does_not_get_copied_to_uncited_list_items_across_blank_lines():
    draft = '**参数 [1]**\n\n- 容量48条。\n- TTL75分钟 [1]。'
    assert mod._review_blocks(draft) == draft.split('\n\n')


async def test_heading_with_a_code_block_retains_the_assignment_guard():
    code = 'per = max(1, pool // max(1, size))'
    draft = '## 原公式\n\n```python\nper = pool // size\n``` [1]'
    result = await mod.AnswerVerifier(model_for([keep(quote=code)])).verify(
        '原公式', draft, [{"number":1,"content":code}])
    assert result.status == 'unsupported' and 'pool // size' not in result.text


def test_consecutive_headings_keep_all_text_and_a_trailing_heading_requires_its_own_review():
    draft = '## 比较\n\n**默认值**\n\n36小时内反馈[1]。\n\n## 无需审批 [1]'
    assert mod._review_blocks(draft) == ['## 比较\n\n**默认值**\n\n36小时内反馈[1]。', '## 无需审批 [1]']


def test_heading_like_text_inside_a_code_fence_is_not_moved_or_split():
    draft = '## 示例\n\n```markdown\n## 默认值\n\n**比较**\n``` [1]'
    assert mod._review_blocks(draft) == [draft]


@pytest.mark.parametrize('title', ['## 无需审批', '## 默认值750分钟', '**默认值** [1]', '## 默认值 [1]',
                                   '## 比较：所有项目均无需审批', '## 各自生成什么（无需审批）'])
def test_headings_with_facts_numbers_references_or_unknown_qualifiers_are_not_layout_labels(title):
    draft = title + '\n\n36小时内反馈[1]。'
    assert mod._review_blocks(draft) == [title, '36小时内反馈[1]。']


def test_captured_missing_heading_indices_were_the_only_omitted_units():
    from pathlib import Path

    case = json.loads((Path(__file__).resolve().parents[2] / 'evaluation/answer_heading_v1.json').read_bytes())['cases'][0]
    blocks, groups = mod._review_units(case['draft'])
    assert len(blocks) == 7 and set(groups) == {2, 3, 4, 5}
    assert blocks[0].startswith('## 各自生成什么\n\n**MQE')
    assert all(blocks[i].startswith('## 比较（仅依据资料）\n\n|') for i in groups)
    # Old model indices cannot be silently re-labelled to approve a new unit.
    with pytest.raises(ValueError):
        mod._apply_review(case['captured_review'], blocks, case['sources'], table_groups=groups)
