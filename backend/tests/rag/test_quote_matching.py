import pytest

from app.rag.quote_matching import quote_is_present


@pytest.mark.parametrize("markup", ["<br>", "<br/>", "<br />", "<BR>"])
def test_docx_break_markup_and_review_newlines_are_equivalent(markup):
    assert quote_is_present('count()\n非缺失值', f'count(){markup}非缺失值')
    assert quote_is_present(f'count(){markup}非缺失值', 'count()\n非缺失值')


@pytest.mark.parametrize("quote,content", [
    ('75分钟', '750分钟'), ('per = n // m', 'per = max(1, n // m)'),
    ('A B', 'A | B'), ('先审批后使用', '先审批……后使用'),
    ('前面后面', '前面<script>后面'), ('<br> ', '任意资料'), ('', '任意资料'),
])
def test_format_normalization_does_not_accept_different_facts_or_empty_markup(quote, content):
    assert not quote_is_present(quote, content)
