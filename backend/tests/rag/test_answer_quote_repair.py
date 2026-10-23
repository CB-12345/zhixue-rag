"""Check quote repair without relaxing source or code validation."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.rag import answer_verifier as mod

DATASET = json.loads((Path(__file__).resolve().parents[1] / 'fixtures/quote_repair.json').read_bytes())


def response(data):
    return SimpleNamespace(content=json.dumps(data))


def keep(index, number=1, quote='容量48条。'):
    return {'index': index, 'keep': True, 'quotes': [{'number': number, 'quote': quote}]}


@pytest.mark.parametrize('case', DATASET['cases'], ids=lambda c: c['id'])
async def test_quote_failures_recover_only_after_fresh_valid_review(case):
    blocks, groups = mod._review_units(case['draft'])
    initial = mod._apply_review(case['captured_review'], blocks, case['sources'], table_groups=groups)
    calls = [response(case['captured_review'])]
    if case['corrected_review']:
        # The retry contains only failed units, renumbered locally.
        outcomes = {}
        mod._apply_review(case['captured_review'], blocks, case['sources'], outcomes, table_groups=groups)
        indices = [r['index'] for r in outcomes['block_results']
                   if r['reason'] in mod._REPAIRABLE_REASONS and mod._citation_numbers(blocks[r['index']])]
        by_index = {d['index']: d for d in case['corrected_review']['blocks']}
        calls.append(response({'blocks': [{**by_index[original], 'index': i}
                                         for i, original in enumerate(indices)]}))
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=calls))
    diagnostics = {}
    result = await mod.AnswerVerifier(model).verify(
        case['query'], case['draft'], case['sources'], diagnostics=diagnostics)
    assert result.status in case['expected_statuses']
    assert all(term in result.text for term in case['required_terms'])
    assert not any(term in result.text for term in case['forbidden_terms'])
    assert model.ainvoke.await_count == len(calls)
    if case['id'].startswith(('q01', 'q02')):
        assert initial.kept_blocks < result.kept_blocks
        assert diagnostics['quote_repair']['restored_indices']
        assert diagnostics['initial_block_results'] != diagnostics['block_results']
    if case['id'].startswith('q01'):
        assert result.repaired_citations == 2
        assert '资料中未说明' not in result.text  # A substantive rejection stays rejected.


async def test_retry_cannot_borrow_an_uncited_sibling_source_or_publish_replacement_text():
    draft = '容量48条[1]。\n\nTTL750分钟[2]。\n\nTTL750分钟[3]。'
    sources = [{'number': 1, 'content': '容量48条。'}, {'number': 2, 'content': 'TTL75分钟。'},
               {'number': 3, 'content': 'TTL750分钟。'}]
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        response({'blocks': [keep(0), keep(1, 2, 'TTL750分钟。'), keep(2, 3, 'TTL750分钟。')]}),
        response({'blocks': [{**keep(0, 3, 'TTL750分钟。'), 'replacement': 'TTL75分钟[2]。'}]}),
    ]))
    result = await mod.AnswerVerifier(model).verify('参数？', draft, sources)
    payload = json.loads(model.ainvoke.call_args.args[0][1].content)
    assert payload['blocks'][0]['allowed_source_numbers'] == [2]
    assert [s['number'] for s in payload['sources']] == [2]
    assert result.kept_blocks == 2 and result.removed_blocks == 1
    assert 'TTL750分钟[2]' not in result.text and 'TTL75分钟' not in result.text


async def test_valid_quote_does_not_allow_an_independent_citation_group_to_lose_support():
    sources = [{'number': 1, 'content': '容量48条。'}, {'number': 2, 'content': 'TTL75分钟。'}]
    first = {'blocks': [keep(0, quote='容量48条！')]}
    retry = {'blocks': [keep(0)]}
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[response(first), response(retry)]))
    result = await mod.AnswerVerifier(model).verify('参数？', '容量48条[1]；TTL750分钟[2]。', sources)
    assert result.status == 'unsupported'
    assert model.ainvoke.await_count == 2


async def test_code_guard_still_applies_after_a_quote_is_repaired():
    sources = [{'number': 1, 'content': 'per = max(1, pool // max(1, size))'}]
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        response({'blocks': [keep(0, quote='per = pool // size')]}),
        response({'blocks': [keep(0, quote=sources[0]['content'])]}),
    ]))
    diagnostics = {}
    result = await mod.AnswerVerifier(model).verify('原公式？', '`per = pool // size` [1]', sources, diagnostics=diagnostics)
    assert result.status == 'unsupported'
    assert diagnostics['quote_repair']['block_results'][0]['reason'] == 'changed_code_assignment'


@pytest.mark.parametrize('bad_retry', [
    {'blocks': []}, {'blocks': [keep(0), keep(0)]},
    {'blocks': [keep(99)]}, {'blocks': [{**keep(0), 'keep': 'true'}]},
])
async def test_bad_retry_protocol_preserves_first_safe_result_and_never_loops(bad_retry):
    sources = [{'number': 1, 'content': '容量48条。TTL75分钟。'}]
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        response({'blocks': [keep(0), keep(1, quote='TTL75分钟！')]}), response(bad_retry),
    ]))
    diagnostics = {}
    result = await mod.AnswerVerifier(model).verify('参数？', '容量48条[1]。\n\nTTL75分钟[1]。', sources, diagnostics=diagnostics)
    assert result.status == 'filtered' and result.kept_blocks == 1 and '容量48条' in result.text
    assert diagnostics['quote_repair']['status'] == 'failed'
    assert model.ainvoke.await_count == 2


async def test_retry_timeout_uses_remaining_budget_and_keeps_verified_content(monkeypatch):
    cancelled = asyncio.Event()
    timeouts = []
    original_wait_for = asyncio.wait_for

    async def record_wait_for(awaitable, timeout):
        timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout)

    async def invoke(*args):
        if len(timeouts) == 1:
            await asyncio.sleep(.02)
            return response({'blocks': [keep(0), keep(1, quote='TTL75分钟！')]})
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(mod.settings, 'ANSWER_VERIFICATION_TIMEOUT_SECONDS', .1)
    monkeypatch.setattr(mod.asyncio, 'wait_for', record_wait_for)
    diagnostics = {}
    result = await mod.AnswerVerifier(SimpleNamespace(ainvoke=invoke)).verify(
        '参数？', '容量48条[1]。\n\nTTL75分钟[1]。', [{'number': 1, 'content': '容量48条。TTL75分钟。'}],
        diagnostics=diagnostics)
    assert result.status == 'filtered' and result.kept_blocks == 1 and cancelled.is_set()
    assert timeouts[0] == .1 and 0 < timeouts[1] < .09
    assert diagnostics['quote_repair']['error_type'] == 'TimeoutError'


async def test_cancellation_during_retry_propagates():
    started = asyncio.Event()

    async def retry(*args):
        started.set()
        await asyncio.Event().wait()

    first = response({'blocks': [keep(0, quote='容量48条！')]})
    count = 0

    async def invoke(*args):
        nonlocal count
        count += 1
        return first if count == 1 else await retry()

    task = asyncio.create_task(mod.AnswerVerifier(SimpleNamespace(ainvoke=invoke)).verify(
        '参数？', '容量48条[1]。', [{'number': 1, 'content': '容量48条。'}]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_no_retry_after_deadline():
    model = SimpleNamespace(ainvoke=AsyncMock())
    data = {'blocks': [keep(0, quote='容量48条！')]}
    diagnostics = {}
    result = await mod.AnswerVerifier(model)._repair_quotes(
        model, {'query': '参数？', 'sources': [{'number': 1, 'content': '容量48条。'}]}, ['容量48条[1]。'],
        data, [{'index': 0, 'reason': 'invalid_quote_or_uncited_source'}],
        asyncio.get_running_loop().time() - 1, diagnostics)
    assert result is data and diagnostics['quote_repair']['status'] == 'budget_exhausted'
    model.ainvoke.assert_not_called()


async def test_retry_provider_exception_does_not_leak_secrets_or_discard_safe_content(monkeypatch):
    warning = Mock()
    monkeypatch.setattr(mod.logger, 'warning', warning)
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        response({'blocks': [keep(0), keep(1, quote='TTL75分钟！')]}), RuntimeError('secret-provider-body'),
    ]))
    diagnostics = {}
    result = await mod.AnswerVerifier(model).verify(
        '参数？', '容量48条[1]。\n\nTTL75分钟[1]。', [{'number': 1, 'content': '容量48条。TTL75分钟。'}],
        diagnostics=diagnostics)
    assert result.status == 'filtered' and result.kept_blocks == 1
    assert 'secret-' not in json.dumps(diagnostics) + str(warning.call_args)


async def test_repaired_table_row_rejoins_previously_verified_rows_in_original_order():
    draft = '| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 [1] |\n| TTL | 75分钟 [1] |'
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        response({'blocks': [keep(0), keep(1, quote='TTL75分钟！')]}),
        response({'blocks': [keep(0, quote='TTL75分钟。')]}),
    ]))
    result = await mod.AnswerVerifier(model).verify('参数？', draft, [{'number': 1, 'content': '容量48条。TTL75分钟。'}])
    assert result.status == 'supported' and result.text == draft and result.kept_blocks == 2
