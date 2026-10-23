"""A model's repeat plan can only remove a verified trailing prose summary."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.rag import answer_verifier as mod

SOURCES = [{'number': 1, 'content': '容量48条。'}, {'number': 2, 'content': 'TTL75分钟。'}]
DRAFT = '容量48条[1]。\n\nTTL75分钟[2]。\n\n因此，容量48条，TTL75分钟[1][2]。'


def review():
    return {'blocks': [
        {'index': 0, 'keep': True, 'quotes': [{'number': 1, 'quote': '容量48条。'}]},
        {'index': 1, 'keep': True, 'quotes': [{'number': 2, 'quote': 'TTL75分钟。'}]},
        {'index': 2, 'keep': True, 'quotes': [{'number': 1, 'quote': '容量48条。'},
                                            {'number': 2, 'quote': 'TTL75分钟。'}], 'redundant_of': [0, 1]},
    ]}


def model_for(*responses):
    return SimpleNamespace(ainvoke=AsyncMock(side_effect=[SimpleNamespace(content=json.dumps(r)) for r in responses]))


async def test_verified_repeat_is_removed_without_misreporting_unsupported_facts():
    diagnostics = {}
    result = await mod.AnswerVerifier(model_for(review())).verify('容量和TTL？', DRAFT, SOURCES, diagnostics=diagnostics)
    assert result.text == '容量48条[1]。\n\nTTL75分钟[2]。'
    assert result.status == 'supported' and result.compacted_blocks == 1 and result.removed_blocks == 0
    assert result.kept_blocks == 2 and mod.OMISSION_NOTICE not in result.text
    assert diagnostics['block_results'][-1]['reason'] == 'compacted_repeat'


@pytest.mark.parametrize('lead', ['因此区别在于：', '即：'])
async def test_captured_comparison_lead_is_recognized_as_a_summary(lead):
    draft = DRAFT.replace('因此，', lead)
    result = await mod.AnswerVerifier(model_for(review())).verify('容量和TTL？', draft, SOURCES)
    assert result.text == '容量48条[1]。\n\nTTL75分钟[2]。'
    assert result.compacted_blocks == 1 and result.removed_blocks == 0


@pytest.mark.parametrize('query', ['请回答后单独总结', '归纳容量与TTL', 'Give a summary', 'Summarize the defaults'])
async def test_explicit_summary_request_preserves_the_requested_summary(query):
    result = await mod.AnswerVerifier(model_for(review())).verify(query, DRAFT, SOURCES)
    assert result.text == DRAFT and result.compacted_blocks == 0


@pytest.mark.parametrize('references', [[2], [99], [-1], [True], [0, 0], ['0'], [], None, [0], [1]])
async def test_invalid_or_incomplete_repeat_plan_cannot_delete_text(references):
    data = review()
    data['blocks'][2]['redundant_of'] = references
    result = await mod.AnswerVerifier(model_for(data)).verify('容量和TTL？', DRAFT, SOURCES)
    assert result.text == DRAFT and result.compacted_blocks == 0


async def test_summary_is_preserved_when_an_antecedent_failed_verification():
    data = review()
    data['blocks'][1] = {'index': 1, 'keep': False, 'quotes': []}
    result = await mod.AnswerVerifier(model_for(data)).verify('容量和TTL？', DRAFT, SOURCES)
    assert '因此，容量48条，TTL75分钟' in result.text and result.compacted_blocks == 0


@pytest.mark.parametrize('last', [
    '另一个所问条件：TTL75分钟[2]。',
    '因此，示例[1]：\n```python\ncapacity = 48\n```',
    '因此，参数如下[1][2]：\n| 参数 | 值 |\n| --- | --- |\n| 容量 | 48条 |',
    '因此，参数[1][2]：\n- 容量48条[1]。',
])
async def test_unrecognized_prose_code_tables_and_lists_cannot_be_compacted(last):
    draft = '容量48条[1]。\n\nTTL75分钟[2]。\n\n' + last
    data = review()
    data['blocks'][2]['quotes'] = [q for q in data['blocks'][2]['quotes'] if q['number'] in mod._citation_numbers(last)]
    result = await mod.AnswerVerifier(model_for(data)).verify('容量和TTL？', draft, SOURCES)
    assert result.compacted_blocks == 0 and last in result.text


async def test_summary_with_a_new_source_cannot_be_compacted():
    data = review()
    data['blocks'][2]['quotes'].append({'number': 3, 'quote': '新增限制。'})
    draft = DRAFT + '新增限制[3]。'
    result = await mod.AnswerVerifier(model_for(data)).verify(
        '容量和TTL？', draft, [*SOURCES, {'number': 3, 'content': '新增限制。'}])
    assert result.text == draft and result.compacted_blocks == 0


async def test_quote_retry_preserves_original_repeat_indices_before_final_compaction():
    first = review()
    first['blocks'][2]['quotes'][0]['quote'] = '容量48条！'
    retry = {'blocks': [{**review()['blocks'][2], 'index': 0, 'redundant_of': []}]}
    model = model_for(first, retry)
    result = await mod.AnswerVerifier(model).verify('容量和TTL？', DRAFT, SOURCES)
    assert result.status == 'supported' and result.compacted_blocks == 1 and '因此' not in result.text
    assert model.ainvoke.await_count == 2
