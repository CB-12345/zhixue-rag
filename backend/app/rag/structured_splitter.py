"""Keep explicit Markdown sections and table rows meaningful within a character budget.

DOCX headings/tables and detected PDF grids are normalized to this representation.
No model calls, inferred headers, cross-page merges or document-wide heading state.
"""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _cells(line):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)]


def _table_start(lines, index):
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    header, separator = _cells(lines[index]), _cells(lines[index + 1])
    return (len(header) >= 2 and len(header) == len(separator)
            and all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator))


def _fence_after(line, fence):
    match = _FENCE.match(line)
    if not match:
        return fence
    marker, tail = match.groups()
    if fence is None:
        return marker
    if marker[0] == fence[0] and len(marker) >= len(fence) and not tail.strip():
        return None
    return fence


def split_structured_document(document, chunk_size, chunk_overlap, separators):
    """Return None for ordinary text so its existing splitting behavior is preserved."""
    sections, headings, body, fence, found = [], [], [], None, False
    for line in document.page_content.splitlines():
        match = _HEADING.match(line) if fence is None else None
        if match:
            if body or headings:
                sections.append((headings.copy(), body))
            level = len(match[1])
            headings = [(depth, value) for depth, value in headings if depth < level]
            headings.append((level, re.sub(r"\s+#+\s*$", "", match[2])))
            body, found = [], True
        else:
            body.append(line)
        fence = _fence_after(line, fence)
    if body or headings:
        sections.append((headings.copy(), body))

    # Only recognize tables outside code fences.
    fence = None
    for _, lines in sections:
        fence = None
        for i, line in enumerate(lines):
            if fence is None and _table_start(lines, i):
                found = True
            fence = _fence_after(line, fence)
    if not found:
        return None

    output = []
    splitters = {}

    def emit(text, prefix, path, kind="text"):
        if not text.strip():
            return
        if len(prefix) >= chunk_size:
            # Oversized context must not consume the entire body budget or lose text.
            text, prefix, kind = prefix + text, "", "structure_fallback"
        budget = chunk_size - len(prefix)
        if budget not in splitters:
            splitters[budget] = RecursiveCharacterTextSplitter(
                chunk_size=budget, chunk_overlap=min(chunk_overlap, budget - 1), separators=[*separators, ""])
        for part in splitters[budget].split_text(text):
            output.append(Document(page_content=prefix + part, metadata={
                **document.metadata, "section_path": [value for _, value in path], "structure_type": kind}))

    for path, lines in sections:
        prefix = "\n".join("#" * level + " " + value for level, value in path)
        prefix = prefix + "\n\n" if prefix else ""
        # Repeating an enormous title would leave no useful space for body text.
        if len(prefix) > chunk_size // 2:
            emit(prefix + "\n".join(lines), "", path, "structure_fallback")
            continue
        if not any(line.strip() for line in lines):
            emit(prefix.strip(), "", path)
            continue
        pending, i, fence = [], 0, None
        while i < len(lines):
            if fence is not None or not _table_start(lines, i):
                pending.append(lines[i])
                fence = _fence_after(lines[i], fence)
                i += 1
                continue
            emit("\n".join(pending), prefix, path)
            pending = []
            header, separator = lines[i], lines[i + 1]
            fields = _cells(header)
            table_prefix = prefix + header + "\n" + separator + "\n"
            i += 2
            rows, had_rows = [], False
            while i < len(lines) and lines[i].strip() and "|" in lines[i] and len(_cells(lines[i])) == len(fields):
                had_rows = True
                row = lines[i]
                if len(table_prefix) + len(row) > chunk_size:
                    if rows:
                        emit("\n".join(rows), table_prefix, path, "table")
                        rows = []
                    values = _cells(row)
                    if len(table_prefix) >= chunk_size // 2 or len(values[0]) >= chunk_size // 3:
                        emit(header + "\n" + separator + "\n" + row, prefix, path, "structure_fallback")
                    else:
                        # Oversized cells use explicit column labels and repeat the row key.
                        identity = f"{fields[0]}：{values[0]}\n"
                        for field, value in zip(fields[1:], values[1:]):
                            emit(value, prefix + identity + field + "：", path, "table_cell")
                else:
                    candidate = "\n".join([*rows, row])
                    if rows and len(table_prefix) + len(candidate) > chunk_size:
                        emit("\n".join(rows), table_prefix, path, "table")
                        rows = []
                    rows.append(row)
                i += 1
            if rows:
                emit("\n".join(rows), table_prefix, path, "table")
            elif not had_rows:
                # A header-only table is still source text and must survive.
                emit(header + "\n" + separator, prefix, path, "table")
        emit("\n".join(pending), prefix, path)
    return output
