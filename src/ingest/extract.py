"""Extraction stage: pull canonical attributes out of every attached source.

Each source is extracted independently and produces its own candidate values.
Candidates are not merged here - that is the conflict resolver's job - because
keeping them separate is what lets the system show a reviewer that the datasheet
and the web page disagreed, and which one it picked.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.ingest.documents import load_source, source_locator
from src.llm.prompts import (
    CLASSIFIER_SYSTEM,
    EXTRACTOR_SYSTEM,
    VISION_SYSTEM,
    classify_prompt,
    extract_prompt,
    vision_extract_prompt,
)
from src.llm.provider import LLMClient, LLMError
from src.models import (
    AttributeValue,
    Method,
    ProductRecord,
    Severity,
    SourceDocument,
    SourceType,
    ValidationIssue,
)
from src.normalize.units import (
    UNIT_FAMILIES,
    awg_to_mm2,
    convert,
    expand_fractions,
    normalize_quantity,
)
from src.schema import GENERIC_CATEGORY, AttributeSchema, AttributeSpec

logger = logging.getLogger(__name__)

# Deterministic patterns worth trying before spending a token on the LLM.
REGEX_RULES: dict[str, re.Pattern[str]] = {
    "ingress_protection": re.compile(r"\b(IP\s?[0-6][0-9K]|IP69K)\b", re.IGNORECASE),
    "unspsc_code": re.compile(r"\bUNSPSC[:\s]*(\d{8})\b", re.IGNORECASE),
    "upc": re.compile(r"\bUPC[:\s]*(\d{12,14})\b", re.IGNORECASE),
    "efficiency_class": re.compile(r"\b(IE[1-5])\b"),
    "insulation_class": re.compile(r"\binsulation\s+class[:\s]+([ABFHN])\b", re.IGNORECASE),
    "tripping_curve": re.compile(r"\b(?:curve|characteristic)[:\s]+([BCDKZ])\b", re.IGNORECASE),
    "bulb_shape": re.compile(
        r"\b(A19|A21|A15|BR30|BR40|PAR16|PAR20|PAR30|PAR38|ST19|ST64|MR16|T8|G25|B11|GU10)\b",
        re.IGNORECASE,
    ),
    "mortar_type": re.compile(r"\btype\s*([NSMO])\b", re.IGNORECASE),
    "board_type": re.compile(r"\b(type\s*x|firelite|fire\s*rated|easi-?lite|lightweight|moisture\s*resistant)\b", re.IGNORECASE),
}

CERT_PATTERN = re.compile(
    r"\b(CE|UL|cUL|CSA|ATEX|IECEx|RoHS|REACH|ISO\s?9001|ISO\s?14001|IEC\s?\d{4,5}"
    r"|EN\s?\d{4,5}|NEMA|VDE|TUV|TÜV|CCC|BIS|EAC|UKCA|SIL\s?[1-4])\b",
    re.IGNORECASE,
)

SOURCE_TYPE_FOR_KIND = {
    "pdf": SourceType.PDF,
    "web": SourceType.WEB,
    "image": SourceType.IMAGE,
    "text": SourceType.INPUT,
    "table": SourceType.INPUT,
}


# --------------------------------------------------------------- classification
def classify(
    record: ProductRecord, schema: AttributeSchema, llm: LLMClient | None
) -> ProductRecord:
    """Settle the product category, escalating to the LLM only when unsure."""
    corpus = " ".join(
        [
            str(record.value_of("product_name", "")),
            str(record.value_of("category", "")),
            str(record.value_of("series", "")),
            record.sku,
            *(s.text[:1500] for s in record.sources if s.text),
        ]
    )
    guess, confidence = schema.guess_category(corpus)

    if confidence >= 0.6 and guess != GENERIC_CATEGORY:
        record.category, record.category_confidence = guess, confidence
        record.log_stage("classify", category=guess, confidence=confidence, method="keyword")
        return record

    if llm is None or not llm.is_configured:
        record.category, record.category_confidence = guess, confidence
        record.log_stage("classify", category=guess, confidence=confidence, method="keyword_only")
        return record

    options = [(k, schema.label_for(k)) for k in schema.category_keys()]
    try:
        parsed, response = llm.complete_json(
            classify_prompt(corpus[:4000], options),
            system=CLASSIFIER_SYSTEM,
            default={},
            max_tokens=300,
        )
    except LLMError as exc:
        logger.warning("Classification LLM call failed for %s: %s", record.sku, exc)
        record.category, record.category_confidence = guess, confidence
        return record

    chosen = (parsed or {}).get("category")
    if chosen in schema.categories:
        record.category = chosen
        record.category_confidence = float((parsed or {}).get("confidence", 0.7))
        record.log_stage(
            "classify",
            category=chosen,
            confidence=record.category_confidence,
            method="llm",
            reason=(parsed or {}).get("reason", ""),
            model=response.model_id,
        )
    else:
        record.category, record.category_confidence = guess, confidence
    return record


def seed_category_attribute(record: ProductRecord, schema: AttributeSchema) -> None:
    """Write the settled category back as an attribute value.

    `category` is a required field in the schema, so leaving the classifier's
    answer only on the record object made every record fail the required-field
    rule for a value the engine had already worked out.
    """
    if not record.category:
        return

    existing = record.attributes.get("category")
    if existing is not None and str(existing.value) == record.category:
        return
    # The supplier wrote "electric motor", the taxonomy says "electric_motor".
    # Exporting the supplier's spelling would break every downstream filter, so
    # the canonical key wins and the original is kept as an alternative.

    stage = next((s for s in reversed(record.stage_log) if s.get("stage") == "classify"), {})
    how = stage.get("method", "keyword")
    record.attributes["category"] = AttributeValue.make(
        key="category",
        value=record.category,
        source_type=SourceType.INFERRED,
        method=Method.LLM_EXTRACT if how == "llm" else Method.KG_INFER,
        source_ref="classifier",
        evidence=(f"classified as {schema.label_for(record.category)} by {how} "
                  f"at {record.category_confidence:.0%} confidence"),
        confidence=max(0.5, min(0.95, record.category_confidence or 0.5)),
        model_id=stage.get("model"),
    )
    if existing is not None and not existing.is_empty:
        record.attributes["category"].alternatives = [
            {
                "value": existing.value,
                "confidence": round(existing.penalised_confidence(), 3),
                "source_type": existing.provenance.source_type.value,
                "method": existing.provenance.method.value,
                "note": "supplier spelling, normalised to the canonical category",
            }
        ]


# ------------------------------------------------------------ regex prepass
def regex_extract(
    text: str, source: SourceDocument, schema: AttributeSchema, category: str | None
) -> dict[str, AttributeValue]:
    """Cheap, high-confidence hits that need no model call."""
    found: dict[str, AttributeValue] = {}
    specs = schema.specs_for(category)
    source_type = SOURCE_TYPE_FOR_KIND.get(source.kind, SourceType.INPUT)

    for key, pattern in REGEX_RULES.items():
        if key not in specs:
            continue
        match = pattern.search(text)
        if not match:
            continue
        value = (match.group(1) if match.groups() else match.group(0)).strip()
        window = text[max(0, match.start() - 60) : match.end() + 60].replace("\n", " ")
        found[key] = AttributeValue.make(
            key=key,
            value=value,
            source_type=source_type,
            method=Method.REGEX,
            source_ref=source_locator(source, window),
            evidence=window.strip(),
        )

    if "certifications" in specs:
        hits = {m.group(0).upper().replace("  ", " ") for m in CERT_PATTERN.finditer(text)}
        if hits:
            found["certifications"] = AttributeValue.make(
                key="certifications",
                value=sorted(hits),
                source_type=source_type,
                method=Method.REGEX,
                source_ref=source.name,
                evidence=", ".join(sorted(hits)),
                confidence=0.85,
            )
    return found


# ------------------------------------------------- deterministic quantities
# Most specification text is "<label> <number> <unit>" in some order, and a
# quarter of a real catalogue is nothing but that. Mining it with the unit
# tables costs nothing, works with no API key at all, and produces values that
# already carry their canonical unit - so this runs before the model and the
# model only sees what is left.

# Key suffixes that merely restate the unit: "power_kw" is written as "power"
# in prose, never as "power kw".
_UNIT_SUFFIXES = {
    "kw", "w", "v", "a", "ka", "kn", "hz", "rpm", "mm", "mm2", "m", "m3h",
    "kg", "c", "bar", "dn", "cv", "months", "pct", "min", "max",
}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_LABEL_GAP = r"[^0-9\n]{0,20}"           # ": ", " rating of ", " = ", " approx "
_TEMP_CUE_RE = re.compile(r"temp|ambient|thermal|°\s*[cf]|deg", re.IGNORECASE)
_TEMP_RANGE_RE = re.compile(
    rf"(-?{_NUM})\s*(?:°|deg\.?|degrees?)?\s*[cC]?\s*(?:\.\.\.|\.\.|to|-|–|~)\s*"
    rf"(\+?-?{_NUM})\s*(?:°|deg\.?|degrees?)?\s*([cCfF])?",
    re.IGNORECASE,
)


def _label_variants(key: str, spec: AttributeSpec) -> list[str]:
    """Text labels that plausibly sit next to this attribute's number."""
    parts = key.split("_")
    while len(parts) > 1 and parts[-1].lower() in _UNIT_SUFFIXES:
        parts.pop()

    label = re.sub(r"\(.*?\)", " ", spec.label).strip().lower()
    variants = {" ".join(parts).lower(), label, *spec.aliases}
    # Two-character labels match everything; they are worse than no label.
    return sorted({v.strip() for v in variants if len(v.strip()) >= 3},
                  key=len, reverse=True)


def _unit_variants(spec: AttributeSpec) -> list[str]:
    """Every spelling of this attribute's unit, longest first so "kw" beats "w"."""
    family = UNIT_FAMILIES.get(spec.unit_family or "", {})
    return sorted(family.get("units", {}), key=len, reverse=True)


def _unique_unit_owners(specs: dict[str, AttributeSpec]) -> dict[str, str]:
    """Unit spellings that can only belong to one attribute in this category.

    "rpm" in a motor record can only be the speed; "mm" could be four different
    dimensions, so it is excluded and left to the labelled pass. Guessing an
    owner for an ambiguous unit is exactly the silent mis-mapping this system
    exists to avoid.
    """
    owners: dict[str, set[str]] = {}
    for key, spec in specs.items():
        for token in _unit_variants(spec):
            owners.setdefault(token.lower(), set()).add(key)
    return {token: next(iter(keys)) for token, keys in owners.items() if len(keys) == 1}


def _numeric_specs(schema: AttributeSchema, category: str | None) -> dict[str, AttributeSpec]:
    return {
        key: spec for key, spec in schema.specs_for(category).items()
        if spec.datatype == "number"
    }


def _build(key: str, spec: AttributeSpec, raw: str, source: SourceDocument,
           source_type: SourceType, window: str, confidence: float) -> AttributeValue | None:
    """Normalise one raw "<number> <unit>" hit into a canonical AttributeValue."""
    value, unit, meta = normalize_quantity(raw, spec.unit_family, spec.unit)
    if value is None:
        return None
    if spec.minimum is not None and value < spec.minimum:
        return None
    if spec.maximum is not None and value > spec.maximum:
        return None

    evidence = window.strip()
    if meta.get("converted"):
        evidence = f"{evidence}  [{meta['from']} converted to {meta['to']}]"

    return AttributeValue.make(
        key=key,
        value=value,
        unit=unit or spec.unit,
        source_type=source_type,
        method=Method.REGEX,
        source_ref=source_locator(source, window),
        evidence=evidence,
        confidence=confidence,
    )


# ------------------------------------------------- trade shorthand patterns
# A distributor description is not prose, it is compressed trade shorthand:
# "3M 775L Stikit Film P150 - Cubitron II 50 Disc/Box" carries the grit, the
# attachment system, the pack quantity and the selling unit in 50 characters.
# These rules decode that shorthand; the LLM never sees what they already got.

_SHAPE_TO_BASE = {
    "MR16": "GU10", "GU10": "GU10",
    "A19": "E26 Medium", "A21": "E26 Medium", "A15": "E26 Medium",
    "BR30": "E26 Medium", "BR40": "E26 Medium", "ST19": "E26 Medium",
    "ST64": "E26 Medium", "PAR30": "E26 Medium", "PAR38": "E26 Medium",
}

_COLOR_SHORTHAND = {
    "blk": "Black", "bk": "Black", "wh": "White", "wht": "White",
    "gry": "Gray", "gy": "Gray", "ss": "Stainless Steel", "sst": "Stainless Steel",
    "bz": "Bronze", "nk": "Nickel", "chr": "Chrome", "alm": "Almond",
}

_UOM_SHORTHAND = {
    "bdl": "BDL", "box": "BX", "bx": "BX", "case": "CS", "cs": "CS",
    "roll": "RL", "rl": "RL", "pk": "PK", "pack": "PK", "ea": "EA",
    "sq": "SQ", "set": "SET",
}


def _first_group(match: re.Match[str]) -> str:
    return (match.group(1) if match.groups() else match.group(0)).strip()


# key -> (pattern, transform). Applied only when the key exists in the category.
DOMAIN_RULES: dict[str, tuple[re.Pattern[str], Any]] = {
    "grit": (
        re.compile(r"\bP\s?(\d{2,4})\b|\b(\d{2,4})\s*(?:grit|grade)\b", re.IGNORECASE),
        lambda m: float(m.group(1) or m.group(2)),
    ),
    "color_temperature_k": (
        # "27k" is trade shorthand for 2700 K; "5000K" is written in full.
        re.compile(r"\b([2-6][05])\s*k\b|\b([2-6]\d{3})\s*k\b", re.IGNORECASE),
        lambda m: float(m.group(1)) * 100 if m.group(1) else float(m.group(2)),
    ),
    "wattage_w": (
        re.compile(r"\b(\d{1,4})\s*w(?:att)?s?\b", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "max_wattage_w": (
        re.compile(r"\b(\d{1,4})\s*w(?:att)?s?\b", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "battery_platform_v": (
        re.compile(r"\b(\d{1,3})\s*v(?:olt)?s?\b(?:\s*max)?", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "battery_voltage_v": (
        re.compile(r"\b(\d{1,3})\s*v(?:olt)?s?\b(?:\s*max)?", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "amp_hours_ah": (
        re.compile(r"\b(\d{1,2}(?:\.\d)?)\s*ah\b", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "steel_gauge": (
        re.compile(r"\b(\d{2})\s*ga(?:uge)?\b", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "pack_quantity": (
        re.compile(
            r"\b(\d{1,4})\s*(?:pc|pcs|pk|pack|ct|count)\b"
            r"|\b(\d{1,4})\s*[a-z]*\s*/\s*(?:box|bx|case|cs|pk|roll)\b",
            re.IGNORECASE,
        ),
        lambda m: float(m.group(1) or m.group(2)),
    ),
    "selling_uom": (
        re.compile(r"\((BDL|BX|CS|RL|PK|EA|SET)\)|/\s*(box|case|roll|pk)\b|\b(\d+)\s*(sq)\b",
                   re.IGNORECASE),
        lambda m: _UOM_SHORTHAND.get(
            (m.group(1) or m.group(2) or m.group(4) or "").lower(), "EA"
        ),
    ),
    "light_count": (
        re.compile(r"\b(\d{1,2})\s*(?:light|lt|lamp)s?\b", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "apparel_size": (
        re.compile(r"\b(XS|S|M|L|XL|2XL|3XL|XXL)\b"),
        lambda m: {"XXL": "2XL"}.get(m.group(1).upper(), m.group(1).upper()),
    ),
    "tooth_count": (
        re.compile(r"\b(\d{1,3})\s*(?:t|tooth|teeth|tpi)\b", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    "attachment_type": (
        re.compile(r"\b(stikit|psa|hook\s*(?:&|and)?\s*loop|velcro|quick\s*change)\b",
                   re.IGNORECASE),
        lambda m: "Stikit / PSA" if m.group(1).lower() in ("stikit", "psa")
        else "Quick Change" if "quick" in m.group(1).lower() else "Hook & Loop",
    ),
    "brushless": (
        re.compile(r"\b(brushless|\bbl\b)\b", re.IGNORECASE),
        lambda m: True,
    ),
    "battery_included": (
        re.compile(r"\b(bare|tool\s*only|kit|starter\s*kit|w/\s*batter)", re.IGNORECASE),
        lambda m: not m.group(1).lower().startswith(("bare", "tool")),
    ),
    "heated": (
        re.compile(r"\bheated\b", re.IGNORECASE),
        lambda m: True,
    ),
    "energy_star": (
        re.compile(r"\benergy\s*star\b", re.IGNORECASE),
        lambda m: True,
    ),
    "fuel_type": (
        re.compile(r"\b(gas|electric|induction|dual\s*fuel|propane)\b", re.IGNORECASE),
        lambda m: m.group(1).title().replace("Dual  Fuel", "Dual Fuel"),
    ),
    "edge_profile": (
        re.compile(r"\b(grooved|square\s*edge|sq\s*edge|scalloped)\b", re.IGNORECASE),
        lambda m: "Grooved Edge" if m.group(1).lower().startswith("groove")
        else "Scalloped" if m.group(1).lower().startswith("scallop") else "Square Edge",
    ),
    "abrasive_form": (
        re.compile(r"\b(belt|flap\s*disc|disc|wheel|sheet|roll|mesh|abranet|pad)\b",
                   re.IGNORECASE),
        lambda m: {
            "belt": "Belt", "disc": "Disc", "wheel": "Disc", "sheet": "Sheet",
            "roll": "Roll", "mesh": "Mesh", "abranet": "Mesh", "pad": "Pad",
        }.get(re.sub(r"\s+", " ", m.group(1).lower()), "Flap Disc"),
    ),
}

# Colour is written in shorthand across every category, so it gets one rule
# applied to whichever colour-ish key the category actually defines.
_COLOR_RE = re.compile(
    r"\b(blk|bk|wh|wht|gry|gy|ss|sst|bz|nk|chr|alm)\b", re.IGNORECASE
)
_COLOR_KEYS = ("color", "device_color", "apparel_color", "finish_color", "color_name")

# "1/2\"x18\"" - width by length. Which attributes those two numbers belong to
# depends on the product class, and nothing else in the description says so.
# Board stock reads thickness x width, then a dash and the stock length in feet:
# "1x6-16'" is a one-inch board, six inches wide, sixteen feet long. Read as a
# plain width-by-length pair it published a one-inch-wide, six-foot board and
# dropped the 16' entirely, because the length slot was already occupied.
BOARD_STOCK_TARGETS: dict[str, tuple[str, str, str]] = {
    "decking_board": ("board_thickness_in", "board_width_in", "board_length_ft"),
    "trim_fascia": ("nominal_thickness_in", "nominal_width_in", "board_length_ft"),
    "lumber_panel": ("nominal_thickness_in", "nominal_width_in", "board_length_ft"),
}

DIMENSION_TARGETS: dict[str, tuple[str, str]] = {
    "abrasive_product": ("abrasive_width_in", "abrasive_length_in"),
    "tape_sealant": ("tape_width_in", "roll_length_ft"),
    "roofing_underlayment": ("roll_width_in", "roll_length_ft"),
    "ceiling_tile": ("tile_width_in", "tile_length_in"),
    "decking_board": ("board_width_in", "board_length_ft"),
    "trim_fascia": ("nominal_width_in", "board_length_ft"),
    "lumber_panel": ("nominal_width_in", "board_length_ft"),
    "drywall_panel": ("panel_width_ft", "panel_length_ft"),
    "metal_roof_panel": ("panel_width_in", "panel_length_ft"),
    "window_unit": ("rough_opening_width_in", "rough_opening_height_in"),
    "skylight": ("rough_opening_width_in", "rough_opening_height_in"),
}

# A disc is quoted as diameter x thickness x arbor, never as width x length:
# "5\"x.045\"x7/8\"" is one product dimension set, not two.
TRIPLE_TARGETS: dict[str, tuple[str, str, str]] = {
    "abrasive_product": ("abrasive_diameter_in", "disc_thickness_in", "arbor_size_in"),
    "saw_blade": ("blade_diameter_in", "kerf_in", "arbor_size_in"),
    # Board stock is quoted the way the yard calls it: "1x6x16" is a one-inch
    # nominal thickness, six inches wide, sixteen feet long. Read as a pair it
    # became a 1-inch-wide board six feet long - wrong on the largest category
    # in the file.
    "decking_board": ("board_thickness_in", "board_width_in", "board_length_ft"),
    "trim_fascia": ("nominal_thickness_in", "nominal_width_in", "board_length_ft"),
    "lumber_panel": ("nominal_thickness_in", "nominal_width_in", "board_length_ft"),
}

# The same rule applies when only two of the three numbers are quoted. A round
# abrasive ("12x1/8 Cut-Off Wheel") gives diameter and thickness; only flat
# stock - belts, rolls, sheets - is quoted width by length. Reading a disc pair
# as width x length publishes a 12-inch width for a 12-inch disc, which the
# judge then has to catch as an internal contradiction.
ROUND_FORM_TARGETS: dict[str, tuple[str, str]] = {
    "abrasive_product": ("abrasive_diameter_in", "disc_thickness_in"),
    "saw_blade": ("blade_diameter_in", "kerf_in"),
}
ROUND_FORMS = {"disc", "flap disc", "wheel"}
_ROUND_CUE_RE = re.compile(r"(disc|wheel|blade)", re.IGNORECASE)

_MARK = r"(\"|''|in\b|'|mm\b)?"
_DIM_TRIPLE_RE = re.compile(
    rf"(?<![\w/])(\d*\.?\d+)\s*{_MARK}\s*[xX×]\s*(\d*\.?\d+)\s*{_MARK}"
    rf"\s*[xX×]\s*(\d*\.?\d+)\s*{_MARK}"
)
_DIM_PAIR_RE = re.compile(
    rf"(?<![\w/])(\d+(?:\.\d+)?)\s*{_MARK}\s*[xX×]\s*(\d+(?:\.\d+)?)\s*{_MARK}"
)
# "1x12-12'" - the trailing segment after the dash is the stock length.
_TRAILING_LENGTH_RE = re.compile(r"[xX×]\s*\d+(?:\.\d+)?\s*-\s*(\d+(?:\.\d+)?)\s*(')?")
_LONE_INCH_RE = re.compile(r"(?<![\w/])(\d+(?:\.\d+)?)\s*(?:\"|''|\bin\b)")

# Categories where a single unqualified inch measurement is the working
# diameter, because that is the number the trade quotes for them.
_LONE_INCH_TARGET = {
    "abrasive_product": "abrasive_diameter_in",
    "saw_blade": "blade_diameter_in",
    "drill_driver_bit": "bit_diameter_in",
    "kitchen_appliance": "appliance_width_in",
    "laundry_appliance": "appliance_width_in",
}


def domain_extract(
    text: str, source: SourceDocument, schema: AttributeSchema, category: str | None
) -> dict[str, AttributeValue]:
    """Decode distributor shorthand into canonical attributes.

    Everything here is deterministic and evidence-carrying: each hit keeps the
    substring it came from, so a reviewer can see why "27k" became 2700 K.
    """
    found: dict[str, AttributeValue] = {}
    if not text.strip():
        return found

    specs = schema.specs_for(category)
    source_type = SOURCE_TYPE_FOR_KIND.get(source.kind, SourceType.INPUT)
    haystack = expand_fractions(text)

    def _emit(key: str, value: Any, match: re.Match[str], confidence: float) -> None:
        spec = specs.get(key)
        if spec is None or key in found:
            return
        if spec.datatype == "number":
            if spec.minimum is not None and float(value) < spec.minimum:
                return
            if spec.maximum is not None and float(value) > spec.maximum:
                return
        if spec.datatype == "enum" and spec.enum and value not in spec.enum:
            return
        window = haystack[max(0, match.start() - 40): match.end() + 40].replace("\n", " ")
        found[key] = AttributeValue.make(
            key=key,
            value=value,
            unit=spec.unit,
            source_type=source_type,
            method=Method.REGEX,
            source_ref=source_locator(source, window),
            evidence=window.strip(),
            confidence=confidence,
        )

    # ---- shorthand rules ---------------------------------------------------
    for key, (pattern, transform) in DOMAIN_RULES.items():
        if key not in specs:
            continue
        match = pattern.search(haystack)
        if match:
            try:
                _emit(key, transform(match), match, 0.88)
            except (TypeError, ValueError):
                continue

    # ---- colour shorthand --------------------------------------------------
    color_key = next((k for k in _COLOR_KEYS if k in specs), None)
    if color_key:
        match = _COLOR_RE.search(haystack)
        if match:
            _emit(color_key, _COLOR_SHORTHAND[match.group(1).lower()], match, 0.80)

    # ---- lamp base inferred from the bulb shape ----------------------------
    if "base_type" in specs and "bulb_shape" in found:
        base = _SHAPE_TO_BASE.get(str(found["bulb_shape"].value).upper())
        shape_match = REGEX_RULES["bulb_shape"].search(haystack)
        if base and shape_match:
            _emit("base_type", base, shape_match, 0.72)
    if "base_type" in specs and "base_type" not in found:
        match = re.search(r"\b(med|medium|cand|candelabra|mog|mogul)\b", haystack, re.IGNORECASE)
        if match:
            token = match.group(1).lower()
            base = ("E26 Medium" if token.startswith("med")
                    else "E12 Candelabra" if token.startswith("cand")
                    else "E39 Mogul")
            _emit("base_type", base, match, 0.80)

    # ---- dimensions: triple first, then pair --------------------------------
    dimension_keys: list[str] = []
    triple = TRIPLE_TARGETS.get(category or "")
    triple_match = _DIM_TRIPLE_RE.search(haystack) if triple else None
    if triple and triple_match:
        numbers = triple_match.groups()
        for key, number, mark in zip(triple, numbers[0::2], numbers[1::2]):
            _emit_dimension(_emit, specs, key, number, mark, triple_match)
            dimension_keys.append(key)

    board = BOARD_STOCK_TARGETS.get(category or "")
    targets = DIMENSION_TARGETS.get(category or "")
    round_targets = ROUND_FORM_TARGETS.get(category or "")
    if round_targets and _is_round_product(found, haystack):
        targets = round_targets

    if board and not triple_match:
        thickness_key, width_key, length_key = board
        match = _DIM_PAIR_RE.search(haystack)
        if match:
            first, first_mark, second, second_mark = match.groups()
            _emit_dimension(_emit, specs, thickness_key, first, first_mark, match)
            _emit_dimension(_emit, specs, width_key, second, second_mark, match)
            dimension_keys += [thickness_key, width_key]

        tail = _TRAILING_LENGTH_RE.search(haystack)
        if tail:
            _emit_dimension(_emit, specs, length_key, tail.group(1), tail.group(2) or "'", tail)
            dimension_keys.append(length_key)
    elif targets and not triple_match:
        width_key, length_key = targets
        match = _DIM_PAIR_RE.search(haystack)
        if match:
            first, first_mark, second, second_mark = match.groups()
            _emit_dimension(_emit, specs, width_key, first, first_mark, match)
            _emit_dimension(_emit, specs, length_key, second, second_mark, match)
            dimension_keys += [width_key, length_key]

        tail = _TRAILING_LENGTH_RE.search(haystack)
        if tail and length_key not in found:
            _emit_dimension(_emit, specs, length_key, tail.group(1), tail.group(2) or "'", tail)
            dimension_keys.append(length_key)

    # ---- lone inch measurement --------------------------------------------
    # Only when no dimension set was found: on "1/2\"x18\" Sanding Belt" the
    # lone-inch rule would otherwise report the belt width as a disc diameter.
    lone_key = _LONE_INCH_TARGET.get(category or "")
    if (lone_key and lone_key in specs and lone_key not in found
            and not any(k in found for k in dimension_keys)):
        match = _LONE_INCH_RE.search(haystack)
        if match:
            _emit(lone_key, float(match.group(1)), match, 0.78)

    return found


def _is_round_product(found: dict[str, AttributeValue], haystack: str) -> bool:
    """True when the description describes a disc, wheel or blade."""
    form = found.get("abrasive_form")
    if form is not None and not form.is_empty:
        return str(form.value).strip().lower() in ROUND_FORMS
    return bool(_ROUND_CUE_RE.search(haystack))


def _emit_dimension(emit: Any, specs: dict[str, AttributeSpec], key: str,
                    number: str, mark: str | None, match: re.Match[str]) -> None:
    """Emit one half of a dimension pair, honouring an explicit foot/inch mark."""
    spec = specs.get(key)
    if spec is None:
        return
    value = float(number)
    token = (mark or "").strip().lower()
    written_unit = ("ft" if token == "'" else
                    "mm" if token == "mm" else
                    "in" if token else None)

    # 2x2 ceiling tiles are feet; 2.75x30 sanding belts are inches. When the
    # description does not say, the attribute's own canonical unit decides.
    if written_unit and spec.unit and written_unit != spec.unit:
        converted, _ = convert(value, written_unit, spec.unit_family or "")
        if converted is not None:
            value = converted
    emit(key, round(value, 4), match, 0.82 if written_unit else 0.74)


def quantity_extract(
    text: str, source: SourceDocument, schema: AttributeSchema, category: str | None
) -> dict[str, AttributeValue]:
    """Pull labelled and unit-anchored numbers out of free text, already converted.

    Two passes, in confidence order:
      1. labelled   "power 1.5 kW", "1.5 kW motor"  - the label proves the owner
      2. unit-only  "1465 rpm"                      - only for units that can
                                                      belong to just one field
    """
    found: dict[str, AttributeValue] = {}
    specs = _numeric_specs(schema, category)
    if not specs:
        return found

    source_type = SOURCE_TYPE_FOR_KIND.get(source.kind, SourceType.INPUT)
    haystack = text.replace(" ", " ")

    # ---- pass 1: label-anchored -------------------------------------------
    for key, spec in specs.items():
        units = _unit_variants(spec)
        unit_alt = "|".join(re.escape(u) for u in units) if units else ""
        for label in _label_variants(key, spec):
            escaped = re.escape(label).replace(r"\ ", r"\s+")
            patterns = [rf"{escaped}{_LABEL_GAP}({_NUM})\s*({unit_alt})?"]
            if unit_alt:
                # "1.5 kW motor" - the unit must be present when the label trails.
                patterns.append(rf"({_NUM})\s*({unit_alt})\s*{escaped}")

            for pattern in patterns:
                match = re.search(pattern, haystack, re.IGNORECASE)
                if not match:
                    continue
                number, unit_token = match.group(1), (match.group(2) or "")
                if not unit_token and spec.unit_family:
                    # A bare number for a unit-bearing field is too weak to trust
                    # unless the field has no unit family at all (counts, poles).
                    continue
                window = haystack[max(0, match.start() - 50): match.end() + 50].replace("\n", " ")
                av = _build(key, spec, f"{number} {unit_token}".strip(), source,
                            source_type, window, confidence=0.90)
                if av:
                    found[key] = av
                    break
            if key in found:
                break

    # ---- pass 2: unit-anchored, only where the unit is unambiguous ---------
    for token, key in _unique_unit_owners(specs).items():
        if key in found or len(token) < 2:
            continue
        spec = specs[key]
        match = re.search(rf"({_NUM})\s*{re.escape(token)}\b", haystack, re.IGNORECASE)
        if not match:
            continue
        window = haystack[max(0, match.start() - 50): match.end() + 50].replace("\n", " ")
        av = _build(key, spec, f"{match.group(1)} {token}", source, source_type,
                    window, confidence=0.75)
        if av:
            found[key] = av

    # ---- special cases ----------------------------------------------------
    if "operating_temp_min_c" in specs and "operating_temp_max_c" in specs:
        match = _TEMP_RANGE_RE.search(haystack)
        window = (haystack[max(0, match.start() - 40): match.end() + 40].replace("\n", " ")
                  if match else "")
        # A bare "7-090" inside a part number matches a numeric range perfectly
        # well, so the range only counts as a temperature when the text says so.
        if match and _TEMP_CUE_RE.search(window) and not {
            "operating_temp_min_c", "operating_temp_max_c"
        } & set(found):
            scale = (match.group(3) or "C").upper()
            for key, group in (("operating_temp_min_c", 1), ("operating_temp_max_c", 2)):
                av = _build(key, specs[key], f"{match.group(group)} {scale}", source,
                            source_type, window, confidence=0.85)
                if av:
                    found[key] = av

    if "conductor_size_mm2" in specs and "conductor_size_mm2" not in found:
        mm2 = awg_to_mm2(haystack)
        if mm2 is not None:
            found["conductor_size_mm2"] = AttributeValue.make(
                key="conductor_size_mm2",
                value=mm2,
                unit="mm2",
                source_type=source_type,
                method=Method.UNIT_CONVERT,
                source_ref=source.name,
                evidence=f"AWG size in text converted to {mm2} mm2",
                confidence=0.85,
            )

    return found


# --------------------------------------------------------------- llm passes
def _known_values_block(record: ProductRecord, limit: int = 12) -> str:
    lines = []
    for key, av in list(record.attributes.items())[:limit]:
        if not av.is_empty:
            lines.append(f"  {key} = {av.value}{(' ' + av.unit) if av.unit else ''}")
    return "\n".join(lines)


def _candidates_from_payload(
    payload: dict[str, Any],
    source: SourceDocument,
    schema: AttributeSchema,
    category: str | None,
    model_id: str,
) -> tuple[dict[str, AttributeValue], list[dict[str, Any]]]:
    """Turn a model JSON response into AttributeValues, keeping the unmapped tail."""
    specs = schema.specs_for(category)
    source_type = SOURCE_TYPE_FOR_KIND.get(source.kind, SourceType.INPUT)
    method = Method.VLM if source.kind == "image" else Method.LLM_EXTRACT

    out: dict[str, AttributeValue] = {}
    for key, item in (payload.get("attributes") or {}).items():
        if key not in specs:
            # The model used a near-miss key; try to rescue it via the alias map.
            resolved, score = schema.resolve_attribute(key, category)
            if resolved is None:
                continue
            key = resolved

        if isinstance(item, dict):
            value = item.get("value")
            evidence = item.get("evidence")
            certain = bool(item.get("certain", True))
            legibility = item.get("legibility")
        else:
            value, evidence, certain, legibility = item, None, True, None

        if value in (None, "", []):
            continue

        av = AttributeValue.make(
            key=key,
            value=value,
            source_type=source_type,
            method=method,
            source_ref=source_locator(source, evidence),
            evidence=evidence,
            model_id=model_id,
        )
        if not certain:
            av.confidence *= 0.7
            av.issues.append(
                ValidationIssue(
                    code="MODEL_UNCERTAIN",
                    severity=Severity.INFO,
                    message="Model flagged this value as uncertain in the source",
                    field_key=key,
                    raised_by="extractor",
                )
            )
        if legibility in ("partial", "poor"):
            av.confidence *= 0.75 if legibility == "partial" else 0.5
        if evidence is None and source.kind != "image":
            # No quotable evidence means the model went outside the source.
            av.confidence *= 0.6
            av.issues.append(
                ValidationIssue(
                    code="NO_EVIDENCE",
                    severity=Severity.WARNING,
                    message="Extracted without a quotable source snippet",
                    field_key=key,
                    raised_by="extractor",
                )
            )
        out[key] = av

    unmapped = payload.get("unmapped") or []
    return out, unmapped if isinstance(unmapped, list) else []


def extract_from_source(
    record: ProductRecord,
    source: SourceDocument,
    schema: AttributeSchema,
    llm: LLMClient,
) -> tuple[dict[str, AttributeValue], list[dict[str, Any]]]:
    """Extract candidate values from one source document."""
    load_source(source)

    if source.kind == "image":
        return _extract_image(record, source, schema, llm)

    text = (source.text or "").strip()
    if len(text) < 20:
        return {}, []

    candidates = regex_extract(text, source, schema, record.category)
    # Narrowest first: named patterns, then trade shorthand, then the
    # generic label/unit miner. Each only fills what the previous left.
    for miner in (domain_extract, quantity_extract):
        for key, av in miner(text, source, schema, record.category).items():
            candidates.setdefault(key, av)

    if llm.is_configured:
        try:
            payload, response = llm.complete_json(
                extract_prompt(
                    source_label=f"{source.kind}: {source.name}",
                    source_text=text,
                    attribute_block=schema.extraction_prompt_block(record.category),
                    known_values=_known_values_block(record),
                ),
                system=EXTRACTOR_SYSTEM,
                default={},
            )
            llm_candidates, unmapped = _candidates_from_payload(
                payload or {}, source, schema, record.category, response.model_id
            )
            # Regex wins ties: it is deterministic and already carries its window.
            for key, av in llm_candidates.items():
                candidates.setdefault(key, av)
            return candidates, unmapped
        except LLMError as exc:
            logger.warning("Extraction failed for %s / %s: %s", record.sku, source.name, exc)

    return candidates, []


def _extract_image(
    record: ProductRecord,
    source: SourceDocument,
    schema: AttributeSchema,
    llm: LLMClient,
) -> tuple[dict[str, AttributeValue], list[dict[str, Any]]]:
    """Vision pass over a product photo or nameplate."""
    if not source.path or not llm.is_configured:
        return {}, []
    if llm.active_provider(vision=True) is None:
        logger.info("No vision-capable provider configured; skipping %s", source.name)
        return {}, []

    context = f"{record.value_of('brand', '')} {record.sku} {record.value_of('product_name', '')}".strip()
    try:
        payload, response = llm.complete_json(
            vision_extract_prompt(schema.extraction_prompt_block(record.category), context),
            system=VISION_SYSTEM,
            images=[source.path],
            default={},
        )
    except LLMError as exc:
        logger.warning("Vision extraction failed for %s: %s", source.name, exc)
        return {}, []

    payload = payload or {}
    # Keep the OCR text on the source so later stages and reviewers can see it.
    visible = payload.get("visible_text") or []
    if visible:
        source.text = "\n".join(str(line) for line in visible)

    if payload.get("image_quality") == "poor":
        record.record_issues.append(
            ValidationIssue(
                code="POOR_IMAGE_QUALITY",
                severity=Severity.INFO,
                message=f"Image {source.name} was too poor to read reliably",
                raised_by="vision",
            )
        )
    return _candidates_from_payload(payload, source, schema, record.category, response.model_id)


def _input_text_source(record: ProductRecord) -> SourceDocument | None:
    """The seeded row values as one text blob, so the miners can read inside them.

    A supplier line like "1.5 kW 4-pole TEFC motor, 400V" carries four
    specifications inside a single mapped cell. Without this the extractors only
    ever see the columns that *failed* to map, and the richest cell in the row -
    the description - is skipped precisely because it mapped cleanly.
    """
    # The SKU is excluded on purpose: a part number is a string of digits and
    # separators that every numeric pattern matches and none of them should.
    parts = [
        f"{key}: {av.value}"
        for key, av in record.attributes.items()
        if isinstance(av.value, str) and len(av.value) >= 4
        and key != "sku" and av.value != record.sku
    ]
    if not parts:
        return None
    return SourceDocument(kind="text", name="input_row_values", text="\n".join(parts))


def extract_all(
    record: ProductRecord, schema: AttributeSchema, llm: LLMClient
) -> dict[str, list[AttributeValue]]:
    """Extract from every source, returning candidates grouped by attribute key.

    The input row values already on the record are seeded in as candidates too,
    so conflict resolution sees them alongside document-derived values.
    """
    grouped: dict[str, list[AttributeValue]] = {}
    for key, av in record.attributes.items():
        grouped.setdefault(key, []).append(av)

    unmapped_all: list[dict[str, Any]] = []
    sources = [*record.sources]
    row_text = _input_text_source(record)
    if row_text is not None:
        sources.append(row_text)

    for source in sources:
        try:
            candidates, unmapped = extract_from_source(record, source, schema, llm)
        except Exception as exc:  # noqa: BLE001 - one bad source must not kill the record
            logger.warning("Source %s failed for %s: %s", source.name, record.sku, exc)
            continue
        for key, av in candidates.items():
            grouped.setdefault(key, []).append(av)
        unmapped_all.extend(unmapped)

    # Try once more to place anything the model could not map to a canonical key.
    for item in unmapped_all:
        name = str(item.get("name", ""))
        key, score = schema.resolve_attribute(name, record.category)
        if key is None or item.get("value") in (None, ""):
            continue
        av = AttributeValue.make(
            key=key,
            value=item["value"],
            source_type=SourceType.INFERRED,
            method=Method.LLM_EXTRACT,
            source_ref="unmapped_recovery",
            evidence=item.get("evidence") or f"{name}: {item['value']}",
            confidence=0.6 * (score / 100.0),
        )
        grouped.setdefault(key, []).append(av)

    record.log_stage(
        "extract",
        sources=len(record.sources),
        keys_with_candidates=len(grouped),
        total_candidates=sum(len(v) for v in grouped.values()),
        unmapped_recovered=len(unmapped_all),
    )
    return grouped
