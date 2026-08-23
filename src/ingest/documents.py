"""Document intelligence: turn PDFs, web pages and images into extractable text.

PDF handling keeps page numbers attached to the text so an extracted value can
cite "datasheet.pdf#page=3" and a reviewer can jump straight to it. Tables are
pulled out separately because spec sheets put most of the real data in them, and
a flattened table loses the row/column pairing that makes a value meaningful.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.models import SourceDocument

logger = logging.getLogger(__name__)

PAGE_MARKER = "\n\n=== PAGE {n} ==="
MAX_CHARS_PER_DOC = 24000          # keeps a single prompt within budget


# --------------------------------------------------------------------- pdf
def extract_pdf(path: str | Path, *, max_pages: int = 30) -> SourceDocument:
    """Extract page-tagged text and tables from a PDF."""
    import fitz  # PyMuPDF

    p = Path(path)
    doc = SourceDocument(kind="pdf", name=p.name, path=str(p))

    try:
        pdf = fitz.open(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot open PDF %s: %s", p, exc)
        doc.text = ""
        return doc

    chunks: list[str] = []
    tables: list[list[list[str]]] = []

    with pdf:
        doc.page_count = pdf.page_count
        for page_no in range(min(pdf.page_count, max_pages)):
            page = pdf[page_no]
            chunks.append(PAGE_MARKER.format(n=page_no + 1))
            chunks.append(page.get_text("text"))

            try:
                found = page.find_tables()
            except Exception:  # noqa: BLE001 - table finder is best-effort
                continue
            for table in found:
                rows = [
                    [(cell or "").strip() for cell in row]
                    for row in table.extract()
                    if any(cell for cell in row)
                ]
                if len(rows) >= 2:
                    tables.append(rows)
                    chunks.append(_render_table(rows, page_no + 1))

    doc.text = _truncate("\n".join(chunks))
    doc.tables = tables
    doc.checksum = doc.fingerprint()
    return doc


def _render_table(rows: list[list[str]], page_no: int) -> str:
    """Render a table as pipe-delimited text so the LLM keeps row/column pairing."""
    lines = [f"\n[TABLE on page {page_no}]"]
    for row in rows:
        lines.append(" | ".join(cell.replace("\n", " ") for cell in row))
    return "\n".join(lines)


def _truncate(text: str, limit: int = MAX_CHARS_PER_DOC) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.7)]
    tail = text[-int(limit * 0.3) :]
    return f"{head}\n\n[... {len(text) - limit} characters omitted ...]\n\n{tail}"


# --------------------------------------------------------------------- web
def extract_web(url: str, *, timeout: float = 20.0) -> SourceDocument:
    """Fetch a product page and reduce it to readable text plus spec tables."""
    import httpx
    from bs4 import BeautifulSoup

    doc = SourceDocument(kind="web", name=url[:120], url=url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fetch failed for %s: %s", url, exc)
        doc.text = ""
        return doc

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    tables: list[list[list[str]]] = []
    table_text: list[str] = []
    for table in soup.find_all("table"):
        rows = [
            [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
            for tr in table.find_all("tr")
        ]
        rows = [r for r in rows if any(r)]
        if len(rows) >= 2:
            tables.append(rows)
            table_text.append(_render_table(rows, 0))

    # Definition lists are the other common home for spec pairs.
    for dl in soup.find_all("dl"):
        pairs = list(zip(dl.find_all("dt"), dl.find_all("dd")))
        if pairs:
            table_text.append(
                "\n[SPEC LIST]\n"
                + "\n".join(
                    f"{dt.get_text(' ', strip=True)} | {dd.get_text(' ', strip=True)}"
                    for dt, dd in pairs
                )
            )

    body = soup.get_text("\n", strip=True)
    body = re.sub(r"\n{3,}", "\n\n", body)
    doc.text = _truncate(body + "\n" + "\n".join(table_text))
    doc.tables = tables
    doc.checksum = doc.fingerprint()
    return doc


# ------------------------------------------------------------------- images
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def prepare_image(path: str | Path, *, max_edge: int = 1568) -> SourceDocument:
    """Validate and downscale an image so it is cheap to send to a VLM."""
    from PIL import Image

    p = Path(path)
    doc = SourceDocument(kind="image", name=p.name, path=str(p))
    if not p.exists() or p.suffix.lower() not in IMAGE_SUFFIXES:
        logger.warning("Not a usable image: %s", p)
        doc.path = None
        return doc

    try:
        with Image.open(p) as img:
            width, height = img.size
            if max(width, height) <= max_edge:
                return doc
            scale = max_edge / max(width, height)
            resized = img.convert("RGB").resize(
                (int(width * scale), int(height * scale)), Image.LANCZOS
            )
            out = p.with_name(f"{p.stem}__vlm{p.suffix if p.suffix != '.webp' else '.jpg'}")
            resized.save(out, quality=88)
            doc.path = str(out)
            doc.name = out.name
    except Exception as exc:  # noqa: BLE001
        logger.warning("Image prep failed for %s: %s", p, exc)
    return doc


# ------------------------------------------------------------------ dispatch
def load_source(source: SourceDocument) -> SourceDocument:
    """Populate a SourceDocument in place based on its kind. Idempotent."""
    if source.text is not None and source.kind in ("pdf", "web", "text", "table"):
        return source

    if source.kind == "pdf" and source.path:
        extracted = extract_pdf(source.path)
        source.text = extracted.text
        source.tables = extracted.tables
        source.page_count = extracted.page_count
    elif source.kind == "web" and source.url:
        extracted = extract_web(source.url)
        source.text = extracted.text
        source.tables = extracted.tables
    elif source.kind == "image" and source.path:
        prepared = prepare_image(source.path)
        source.path = prepared.path
        source.name = prepared.name

    source.checksum = source.fingerprint()
    return source


def source_locator(source: SourceDocument, evidence: str | None) -> str:
    """Build a citable reference like 'datasheet.pdf#page=3' from an evidence quote."""
    base = source.name
    if not evidence or not source.text:
        return base
    position = source.text.find(evidence[:60])
    if position == -1:
        return base
    if source.kind == "pdf":
        page = source.text.count("=== PAGE ", 0, position)
        return f"{base}#page={max(page, 1)}"
    return f"{base}#offset={position}"
