"""Match continuous evidence text while tolerating document line-break markup."""
import re

_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def quote_is_present(quote: str, content: str) -> bool:
    # DOCX code/table cells use <br>; reviewers often return ordinary newlines.
    # Keep punctuation, pipes, numbers and operators: this is not fuzzy matching.
    def normalized(text):
        return "".join(_BREAK.sub("\n", text).split()).casefold()

    needle = normalized(quote)
    return bool(needle) and needle in normalized(content)
