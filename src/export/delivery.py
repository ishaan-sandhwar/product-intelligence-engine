"""Render golden records into the Unilog delivery template.

The template's header row *is* the contract, so it is read from the CSV rather
than hard-coded: a revised template drops in without a code change. Three things
make this more than a column rename:

  1. taxonomy   the pipeline settles one Fine class; the template wants
                Dept / Class / Fine / Classpath, which the schema derives.
  2. attributes the template stores specifications as 60 generic
                LABEL / VALUE / UOM triplets, not as named columns, so the
                canonical attributes are packed into slots in priority order.
  3. content    six description variants, each with its own register, plus 20
                feature slots - all generated against the verified attributes.

Anything the engine could not verify is left blank on purpose. A blank cell is
recoverable; a confident wrong value is not.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import (
    DELIVERY_ATTRIBUTE_SLOTS,
    DELIVERY_FEATURE_SLOTS,
    DELIVERY_TEMPLATE_PATH,
    MIN_PUBLISH_CONFIDENCE,
)
from src.models import AttributeValue, ProductRecord
from src.schema import AttributeSchema, AttributeSpec

logger = logging.getLogger(__name__)

# Canonical keys that already have a dedicated column: they must not also be
# spent on one of the limited attribute triplet slots.
DEDICATED_KEYS = {
    "sku", "product_name", "brand", "manufacturer", "category", "certifications",
    "warranty_months", "upc", "unspsc_code", "country_of_origin", "discontinued",
    "pack_quantity", "selling_uom", "length_in", "width_in", "height_in", "weight_lb",
}

# Document columns are matched by keyword against the attached source filenames.
DOCUMENT_COLUMNS = {
    "Specification Sheet": ("spec", "specification", "datasheet", "data sheet"),
    "Instruction/Installation Manual": ("install", "instruction"),
    "Service Manual": ("service",),
    "Owners/User Manual": ("owner", "user manual", "manual"),
    "Warranty Information": ("warranty",),
    "SDS": ("sds", "msds", "safety data"),
    "Line Drawing": ("line drawing", "drawing", "dimension"),
    "Catalog": ("catalog", "catalogue"),
    "Submittal": ("submittal",),
    "Technical Bulletin": ("bulletin",),
}

_INDEXED = {
    "ref_url": re.compile(r"^Ref URL (\d+)$", re.IGNORECASE),
    "feature": re.compile(r"^ITEM_FEATURES_(\d+)$", re.IGNORECASE),
    "attr_label": re.compile(r"^ATTRIBUTE_LABEL (\d+)$", re.IGNORECASE),
    "attr_value": re.compile(r"^ATTRIBUTE_VALUE (\d+)$", re.IGNORECASE),
    "attr_uom": re.compile(r"^ATTRIBUTE_UOM (\d+)$", re.IGNORECASE),
    "alt_image": re.compile(r"^Alternate Image (\d+)$", re.IGNORECASE),
}


# ------------------------------------------------------------------ template
def load_template_columns(path: Path | str = DELIVERY_TEMPLATE_PATH) -> list[str]:
    """Read the delivery template's header row, in order."""
    target = Path(path)
    if not target.exists():
        logger.warning("Delivery template not found at %s; using the built-in header", target)
        return _fallback_columns()

    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    columns = [c.strip() for c in header if c.strip()]
    logger.info("Delivery template: %d columns from %s", len(columns), target.name)
    return columns or _fallback_columns()


def _fallback_columns() -> list[str]:
    """The template's shape, for when the file itself is not available."""
    columns = ["MFR URL"] + [f"Ref URL {i}" for i in range(1, 6)] + [
        "PART_NUMBER", "Dept", "Class", "Fine", "SKU - MY_PART_NUMBER", "Mfg_Part_Num",
        "Part_Desc", "E1_Brand", "Unilog_Brand", "DIB_Brand", "Part_Manuf",
        "MANUFACTURER_NAME", "BRAND_NAME", "TRADE_NAME", "MANUFACTURER_PART_NUMBER",
        "ALTERNATE_PART_NUMBER", "Classpath", "MOBILE_DESC", "INVOICE_DESC",
        "SHORT_DESC", "LONG_DESC1", "RETAIL_DESC", "MARKETING_DESCRIPTION",
    ]
    columns += [f"ITEM_FEATURES_{i}" for i in range(1, DELIVERY_FEATURE_SLOTS + 1)]
    columns += ["With", "Standard/Approvals", "Prop 65", "Application", "Includes",
                "Product Name"]
    for i in range(1, DELIVERY_ATTRIBUTE_SLOTS + 1):
        columns += [f"ATTRIBUTE_LABEL {i}", f"ATTRIBUTE_VALUE {i}", f"ATTRIBUTE_UOM {i}"]
    columns += ["UPC", "EAN", "GTIN", "UNSPSC", "Warranty", "List Price", "Selling Qty",
                "Selling UOM", "Standard Packaging Information",
                "LENGTH", "LENGTH_UOM", "HEIGHT", "HEIGHT_UOM", "WIDTH", "WIDTH_UOM",
                "WEIGHT", "WEIGHT_UOM", "VOLUME", "VOLUME_UOM", "Product Image"]
    columns += [f"Alternate Image {i}" for i in range(1, 5)]
    columns += list(DOCUMENT_COLUMNS) + ["Video Link 1", "Video Link 2",
                                         "Country Of Origin", "Discontinued",
                                         "Actual Image (Yes/No)"]
    return columns


# ------------------------------------------------------------------ helpers
def _fmt(value: Any) -> str:
    """Render one value the way a PIM import expects to read it."""
    if value is None or value == "" or value == []:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value if v not in (None, ""))
    return str(value).strip()


def _content(record: ProductRecord, key: str) -> str:
    av = record.content.get(key)
    return "" if av is None or av.is_empty else _fmt(av.value)


def _input_cell(record: ProductRecord, header: str) -> str:
    """The value that arrived in a named input column, exactly as supplied.

    Three brand columns collapse onto one canonical `brand` and the placeholder
    markers are dropped at ingest, but the delivery file is expected to carry
    the supplied columns back unchanged - so they are read from the raw row the
    loader kept on the ingest stage entry rather than reconstructed.
    """
    for stage in record.stage_log:
        if stage.get("stage") == "ingest" and isinstance(stage.get("raw_row"), dict):
            raw = stage["raw_row"]
            if header in raw:
                return _fmt(raw[header])
            lowered = {str(k).lower(): v for k, v in raw.items()}
            return _fmt(lowered.get(header.lower(), ""))
    return ""


def _publishable(record: ProductRecord, schema: AttributeSchema,
                 min_confidence: float) -> list[tuple[str, AttributeSpec, AttributeValue]]:
    """Attributes worth a triplet slot, most important first.

    Ordering is required fields, then schema weight, then confidence - the
    template has 60 slots and a rich record can exceed them, so what gets cut
    has to be the least useful thing, not whatever hashed last.
    """
    specs = schema.specs_for(record.category)
    rows: list[tuple[str, AttributeSpec, AttributeValue]] = []

    for key, av in record.attributes.items():
        if key in DEDICATED_KEYS or av.is_empty:
            continue
        if av.penalised_confidence() < min_confidence:
            continue
        spec = specs.get(key)
        if spec is None:
            continue
        rows.append((key, spec, av))

    rows.sort(key=lambda row: (
        not row[1].required,
        -row[1].weight,
        -row[2].penalised_confidence(),
        row[0],
    ))
    return rows


def _sources_by_kind(record: ProductRecord, kind: str) -> list[str]:
    return [s.url or s.name for s in record.sources if s.kind == kind]


def _warranty_text(record: ProductRecord) -> str:
    months = record.value_of("warranty_months")
    if months in (None, ""):
        return ""
    months = float(months)
    if months >= 12 and months % 12 == 0:
        years = int(months // 12)
        return f"{years} Year{'s' if years > 1 else ''} Limited"
    return f"{months:g} Month Limited"


# --------------------------------------------------------------------- rows
def build_row(record: ProductRecord, schema: AttributeSchema, *,
              min_confidence: float = MIN_PUBLISH_CONFIDENCE) -> dict[str, str]:
    """Map one golden record onto the delivery template's columns."""
    web_sources = _sources_by_kind(record, "web")
    images = _sources_by_kind(record, "image")
    documents = [s.name for s in record.sources if s.kind == "pdf"]

    row: dict[str, str] = {
        "MFR URL": web_sources[0] if web_sources else "",
        "PART_NUMBER": record.sku,
        "Dept": schema.dept_for(record.category),
        "Class": schema.class_for(record.category),
        "Fine": schema.fine_for(record.category),
        "SKU - MY_PART_NUMBER": "",
        "Mfg_Part_Num": record.sku,
        "Part_Desc": _input_cell(record, "Part_Desc")
                     or _fmt(record.value_of("product_name")),
        "E1_Brand": _input_cell(record, "E1_Brand"),
        "Unilog_Brand": _input_cell(record, "Unilog_Brand"),
        "DIB_Brand": _input_cell(record, "DIB_Brand"),
        "Part_Manuf": _input_cell(record, "Part_Manuf")
                      or _fmt(record.value_of("manufacturer")),
        "MANUFACTURER_NAME": _fmt(record.value_of("manufacturer")),
        "BRAND_NAME": _fmt(record.value_of("brand")),
        "TRADE_NAME": "",
        "MANUFACTURER_PART_NUMBER": record.sku,
        "ALTERNATE_PART_NUMBER": "",
        "Classpath": schema.classpath_for(record.category),
        "MOBILE_DESC": _content(record, "mobile_desc"),
        "INVOICE_DESC": _content(record, "invoice_desc"),
        "SHORT_DESC": _content(record, "short_desc"),
        "LONG_DESC1": _content(record, "long_desc"),
        "LONG_DESC": _content(record, "long_desc"),
        "RETAIL_DESC": _content(record, "retail_desc"),
        "MARKETING_DESCRIPTION": _content(record, "marketing_description"),
        "With": _content(record, "with_clause"),
        "Includes": _content(record, "includes"),
        "Application": _content(record, "application"),
        "Standard/Approvals": " | ".join(
            _fmt(v) for v in (record.value_of("certifications") or [])
        ) if isinstance(record.value_of("certifications"), list)
          else _fmt(record.value_of("certifications")),
        "Prop 65": "",
        "Product Name": _content(record, "product_name_generic")
                        or schema.fine_for(record.category),
        "UPC": _fmt(record.value_of("upc")),
        "EAN": "",
        "GTIN": _fmt(record.value_of("upc")),
        "UNSPSC": _fmt(record.value_of("unspsc_code")),
        "Warranty": _warranty_text(record),
        "List Price": "",
        "Selling Qty": _fmt(record.value_of("pack_quantity")),
        "Selling UOM": _fmt(record.value_of("selling_uom")),
        "Standard Packaging Information": "",
        "LENGTH": _fmt(record.value_of("length_in")),
        "LENGTH_UOM": "in" if record.value_of("length_in") is not None else "",
        "WIDTH": _fmt(record.value_of("width_in")),
        "WIDTH_UOM": "in" if record.value_of("width_in") is not None else "",
        "HEIGHT": _fmt(record.value_of("height_in")),
        "HEIGHT_UOM": "in" if record.value_of("height_in") is not None else "",
        "WEIGHT": _fmt(record.value_of("weight_lb")),
        "WEIGHT_UOM": "lb" if record.value_of("weight_lb") is not None else "",
        "VOLUME": "",
        "VOLUME_UOM": "",
        "Product Image": images[0] if images else "",
        "Country Of Origin": _fmt(record.value_of("country_of_origin")),
        "Discontinued": _fmt(record.value_of("discontinued")),
        "Actual Image (Yes/No)": "Yes" if images else "",
        "Video Link 1": "",
        "Video Link 2": "",
    }

    for index, url in enumerate(web_sources[1:6], start=1):
        row[f"Ref URL {index}"] = url
    for index, image in enumerate(images[1:5], start=1):
        row[f"Alternate Image {index}"] = image

    for column, hints in DOCUMENT_COLUMNS.items():
        match = next((d for d in documents if any(h in d.lower() for h in hints)), "")
        row[column] = match

    features = record.content.get("item_features")
    if features is not None and isinstance(features.value, list):
        for index, feature in enumerate(features.value[:DELIVERY_FEATURE_SLOTS], start=1):
            row[f"ITEM_FEATURES_{index}"] = _fmt(feature)

    for index, (_, spec, av) in enumerate(
        _publishable(record, schema, min_confidence)[:DELIVERY_ATTRIBUTE_SLOTS], start=1
    ):
        row[f"ATTRIBUTE_LABEL {index}"] = spec.label
        row[f"ATTRIBUTE_VALUE {index}"] = _fmt(av.value)
        row[f"ATTRIBUTE_UOM {index}"] = spec.unit or ""

    return row


def to_delivery_frame(
    records: Iterable[ProductRecord],
    schema: AttributeSchema,
    *,
    columns: list[str] | None = None,
    min_confidence: float = MIN_PUBLISH_CONFIDENCE,
) -> pd.DataFrame:
    """Build the full delivery frame, in template column order."""
    columns = columns or load_template_columns()
    rows = [build_row(r, schema, min_confidence=min_confidence) for r in records]
    # reindex adds every template column the rows never touched, in one pass and
    # in template order.
    return pd.DataFrame(rows).reindex(columns=columns).fillna("")


def write_delivery_csv(
    records: Iterable[ProductRecord],
    schema: AttributeSchema,
    path: Path | str,
    *,
    min_confidence: float = MIN_PUBLISH_CONFIDENCE,
) -> Path:
    """Write the delivery CSV and return where it landed."""
    frame = to_delivery_frame(records, schema, min_confidence=min_confidence)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    logger.info("Delivery export: %d rows x %d columns -> %s",
                len(frame), len(frame.columns), target)
    return target


def coverage_report(frame: pd.DataFrame) -> dict[str, Any]:
    """How much of the template the export actually fills, by block."""
    def _filled(columns: list[str]) -> float:
        if not columns or frame.empty:
            return 0.0
        subset = frame[columns]
        return round(100.0 * (subset.astype(str).apply(lambda c: c.str.strip() != "")).mean().mean(), 1)

    attribute_cols = [c for c in frame.columns if _INDEXED["attr_value"].match(c)]
    feature_cols = [c for c in frame.columns if _INDEXED["feature"].match(c)]
    content_cols = [c for c in ("MOBILE_DESC", "INVOICE_DESC", "SHORT_DESC", "LONG_DESC1",
                                "RETAIL_DESC", "MARKETING_DESCRIPTION") if c in frame.columns]
    identity_cols = [c for c in ("PART_NUMBER", "Mfg_Part_Num", "MANUFACTURER_NAME",
                                 "BRAND_NAME", "Part_Desc") if c in frame.columns]
    taxonomy_cols = [c for c in ("Dept", "Class", "Fine", "Classpath") if c in frame.columns]

    filled_slots = (
        frame[attribute_cols].astype(str).apply(lambda c: c.str.strip() != "").sum(axis=1)
        if attribute_cols else pd.Series([0] * len(frame))
    )
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "identity_pct": _filled(identity_cols),
        "taxonomy_pct": _filled(taxonomy_cols),
        "content_pct": _filled(content_cols),
        "attribute_slots_used_avg": round(float(filled_slots.mean()), 1) if len(frame) else 0.0,
        "attribute_slots_used_max": int(filled_slots.max()) if len(frame) else 0,
        "feature_slots_used_avg": round(
            float(frame[feature_cols].astype(str).apply(lambda c: c.str.strip() != "").sum(axis=1).mean()), 1
        ) if feature_cols and len(frame) else 0.0,
        "overall_pct": _filled(list(frame.columns)),
    }
