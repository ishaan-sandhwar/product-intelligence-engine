"""Reads the input catalogue and turns each row into a seeded ProductRecord.

Input is deliberately forgiving: any CSV / Excel / JSON / JSONL with a SKU-ish
column works. Column headers are mapped onto canonical attribute keys through
the alias dictionary, and anything that does not map is kept as an unmapped
source note rather than silently dropped.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.models import (
    AttributeValue,
    Method,
    ProductRecord,
    SourceDocument,
    SourceType,
)
from src.schema import AttributeSchema, load_schema

logger = logging.getLogger(__name__)

SKU_HINTS = (
    "sku", "part number", "part num", "part_no", "partno", "part no", "mfg part",
    "mpn", "item code", "article", "catalog", "catalogue", "model",
    "product code", "product_id", "id",
)

# Distributor exports use explicit "nothing here" markers rather than blanks.
# Loading "-- Unbranded --" as the brand would look like data and score like
# data, so these are treated as empty at the door.
PLACEHOLDER_VALUES = {
    "n/a", "na", "none", "null", "nan", "unknown", "unspecified", "tbd",
    "-", "--", "---", "not applicable", "no brand",
}
_PLACEHOLDER_RE = re.compile(r"^\s*--.*--\s*$")

# "Freud Inc (2435)" - the trailing bracket is the distributor's vendor code,
# not part of the manufacturer name.
_VENDOR_CODE_RE = re.compile(r"\s*\(([A-Z0-9]{3,8})\)\s*$")

# Columns that describe an attached asset rather than an attribute value.
ASSET_HINTS = {
    "pdf": ("datasheet", "spec sheet", "specsheet", "pdf", "manual", "document", "brochure"),
    "image": ("image", "photo", "picture", "img", "thumbnail", "asset"),
    "web": ("url", "link", "webpage", "product page", "website", "source url"),
}


def _norm_header(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def _detect_sku_column(columns: Iterable[str]) -> str | None:
    normalised = {c: _norm_header(c) for c in columns}
    for col, low in normalised.items():
        if low in ("sku", "part number", "mpn"):
            return col
    for col, low in normalised.items():
        if any(hint in low for hint in SKU_HINTS):
            return col
    return None


def is_placeholder(value: str) -> bool:
    """True when a cell carries a 'no value' marker rather than a value."""
    stripped = value.strip().lower()
    return (not stripped
            or stripped in PLACEHOLDER_VALUES
            or bool(_PLACEHOLDER_RE.match(stripped)))


def clean_cell(key: str, value: str) -> tuple[str, str | None]:
    """Normalise one mapped cell. Returns (value, vendor_code_if_stripped)."""
    if key in ("manufacturer", "brand", "supplier"):
        match = _VENDOR_CODE_RE.search(value)
        if match:
            return _VENDOR_CODE_RE.sub("", value).strip(), match.group(1)
    return value.strip(), None


def _classify_asset_column(header: str) -> str | None:
    low = _norm_header(header)
    for kind, hints in ASSET_HINTS.items():
        if any(hint in low for hint in hints):
            return kind
    return None


def read_table(path: str | Path) -> pd.DataFrame:
    """Load CSV / Excel / JSON / JSONL into a DataFrame with string-safe cells."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in (".csv", ".txt", ".tsv"):
        sep = "\t" if suffix == ".tsv" else None
        frame = pd.read_csv(p, sep=sep, engine="python", dtype=str, keep_default_na=False)
    elif suffix in (".xlsx", ".xls", ".xlsm"):
        frame = pd.read_excel(p, dtype=str).fillna("")
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(rows).fillna("")
    elif suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # Accept {"products": [...]} or a dict keyed by SKU.
            payload = payload.get("products") or payload.get("items") or list(payload.values())
        frame = pd.DataFrame(payload).fillna("")
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def row_to_record(
    row: dict[str, Any],
    schema: AttributeSchema,
    *,
    source_name: str = "input_catalog",
    row_index: int = 0,
) -> ProductRecord:
    """Seed a ProductRecord from one input row, mapping headers to canonical keys."""
    sku_col = _detect_sku_column(row.keys())
    sku = str(row.get(sku_col, "")).strip() if sku_col else ""
    if not sku:
        sku = f"UNKNOWN-{row_index:05d}"

    # First pass over free text so the category guess can use it.
    text_blob = " ".join(str(v) for v in row.values() if v)
    category, cat_conf = schema.guess_category(text_blob)

    record = ProductRecord(sku=sku, category=category, category_confidence=cat_conf)
    unmapped: dict[str, str] = {}

    for header, raw in row.items():
        value = str(raw).strip() if raw is not None else ""
        if is_placeholder(value):
            continue

        asset_kind = _classify_asset_column(header)
        if asset_kind:
            record.sources.append(
                SourceDocument(
                    kind=asset_kind,
                    name=f"{header}: {value[:80]}",
                    path=value if asset_kind in ("pdf", "image") else None,
                    url=value if asset_kind == "web" else None,
                )
            )
            continue

        key, score = schema.resolve_attribute(header, category)
        if key is None:
            unmapped[header] = value
            continue

        cleaned, vendor_code = clean_cell(key, value)
        if key in record.attributes and not record.attributes[key].is_empty:
            # Several columns can map to the same canonical key (three brand
            # columns in a distributor export). First non-placeholder wins; the
            # rest are kept as alternatives so the choice stays auditable.
            record.attributes[key].alternatives.append(
                {"value": cleaned, "source_ref": header, "method": "input_field"}
            )
            continue

        record.attributes[key] = AttributeValue.make(
            key=key,
            value=cleaned,
            source_type=SourceType.INPUT,
            method=Method.INPUT_FIELD,
            source_ref=f"{source_name}#row={row_index}&col={header}",
            evidence=f"{header}: {value}",
            confidence=0.95 if score >= 100 else 0.95 * (score / 100.0),
        )
        if vendor_code:
            record.attributes[key].alternatives.append(
                {"value": value, "note": f"vendor code {vendor_code} stripped"}
            )

    # Keep unmapped columns as a text source so extraction can still mine them.
    if unmapped:
        blob = "\n".join(f"{k}: {v}" for k, v in unmapped.items())
        record.sources.append(
            SourceDocument(
                kind="text",
                name=f"unmapped_columns_row_{row_index}",
                text=blob,
            )
        )

    record.log_stage(
        "ingest",
        source=source_name,
        row=row_index,
        mapped=len(record.attributes),
        unmapped=len(unmapped),
        assets=len(record.sources),
        # The delivery file has to echo the supplied columns back verbatim,
        # including the "-- Unbranded --" markers the pipeline treats as empty,
        # so the row is kept exactly as it arrived.
        raw_row={k: str(v).strip() for k, v in row.items() if str(v).strip()},
    )
    return record


def load_catalog(
    path: str | Path,
    *,
    limit: int | None = None,
    schema: AttributeSchema | None = None,
) -> list[ProductRecord]:
    """Load an input file into seeded ProductRecords."""
    schema = schema or load_schema()
    frame = read_table(path)
    if limit:
        frame = frame.head(limit)

    records: list[ProductRecord] = []
    seen: dict[str, int] = {}
    for idx, row in enumerate(frame.to_dict(orient="records")):
        record = row_to_record(row, schema, source_name=Path(path).name, row_index=idx)
        # De-duplicate SKUs by suffixing, so nothing is silently overwritten.
        if record.sku in seen:
            seen[record.sku] += 1
            record.sku = f"{record.sku}--dup{seen[record.sku]}"
        else:
            seen[record.sku] = 0
        records.append(record)

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def attach_assets(
    records: list[ProductRecord], asset_dir: str | Path, *, by_sku: bool = True
) -> int:
    """Attach PDFs and images from a folder to records by SKU-in-filename match.

    Returns the number of assets attached.
    """
    folder = Path(asset_dir)
    if not folder.exists():
        return 0

    index = {r.sku.upper().replace(" ", ""): r for r in records}
    attached = 0
    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        suffix = file.suffix.lower()
        if suffix == ".pdf":
            kind = "pdf"
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            kind = "image"
        else:
            continue

        stem = file.stem.upper().replace(" ", "").replace("_", "-")
        target = None
        if by_sku:
            for sku_key, record in index.items():
                if sku_key and sku_key in stem:
                    target = record
                    break
        if target is None:
            continue

        target.sources.append(
            SourceDocument(kind=kind, name=file.name, path=str(file))
        )
        attached += 1

    logger.info("Attached %d assets from %s", attached, folder)
    return attached
