"""Extract text-bearing PDF grids with explicit columns; leave other pages as text."""
from threading import Lock

from app.core.logger_handler import logger

_TABLE_LOCK = Lock()  # PyMuPDF's table finder uses process-global working arrays.


def _cell_text(value):
    return (value or "").strip().replace("|", "\\|").replace("\n", "<br>")


def extract_page_text(page):
    plain = page.get_text().strip()
    if not plain:
        return plain
    try:
        with _TABLE_LOCK:
            return _extract_tables(page, plain)
    except Exception as exc:
        logger.warning(f"PDF 表格结构提取不可用，保留原文字: {type(exc).__name__}")
        return plain


def _extract_tables(page, plain):
    tables = []
    for table in page.find_tables(strategy="lines_strict").tables:
        rows = table.extract()
        if not rows or len(rows) < 2 or len(rows[0]) < 2:
            continue
        if any(len(row) != len(rows[0]) for row in rows):
            continue
        # Merged cells are ambiguous; preserve their original text representation.
        if any(cell is None for row in rows for cell in row):
            continue
        if table.header.external:
            names = table.header.names
            if len(names) != len(rows[0]) or not all(names):
                continue
            rows = [names, *rows]
        lines = ["| " + " | ".join(_cell_text(cell) for cell in row) + " |" for row in rows]
        lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
        tables.append((table.bbox, "\n".join(lines)))
    if not tables:
        return plain
    items = [(bbox[1], bbox[0], text) for bbox, text in tables]
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = []
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                x, y = (x0 + x1) / 2, (y0 + y1) / 2
                if not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b, _ in tables):
                    spans.append(span["text"])
            if spans:
                items.append((line["bbox"][1], line["bbox"][0], "".join(spans)))
    return "\n\n".join(text for _, _, text in sorted(items))
