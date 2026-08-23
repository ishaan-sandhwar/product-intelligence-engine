"""Brand, manufacturer and supplier resolution for distributor exports.

A distributor export names the *account it buys from*, not the company that made
the product. `Part_Manuf` holds "Jam Industrial Supply LLC" for a 3M abrasive and
"Freud Inc" for a Diablo sanding belt - the same column, two different kinds of
company. Publishing that column as the manufacturer puts a distributor's name on
the manufacturer line of every one of those records, and leaves `brand` empty on
the 554 rows whose brand columns carry only "-- Unbranded --" markers.

Empty brand has a second cost. Peer inference groups a brandless record with the
whole category, so a Diablo belt sitting in a 3M-dominated abrasives group
inherits 3M's series. Filling brand deterministically is what keeps that group
honest; `ProductGraph.infer_from_peers` enforces the boundary.

Resolution order, most trustworthy first:
  brand         input columns -> description scan -> supplier name
  manufacturer  supplier (when the account is a maker) -> brand
  supplier      the input column, verbatim
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

from src.models import (
    AttributeValue,
    Method,
    ProductRecord,
    Severity,
    SourceType,
    ValidationIssue,
)
from src.schema import AttributeSchema

logger = logging.getLogger(__name__)

# Trade brands that appear inside a description but never in a brand column.
# Aliases map a sub-brand or product line back to the brand that owns it, so
# "Cubitron II" and "Stikit" both resolve to 3M rather than becoming brands.
SEED_BRANDS: dict[str, tuple[str, ...]] = {
    "3M": ("3m", "cubitron", "stikit", "hookit", "trizact", "scotch-brite"),
    "Diablo": ("diablo",),
    "DeWalt": ("dewalt", "de walt", "dewlt"),
    "Milwaukee": ("milwaukee",),
    "Makita": ("makita",),
    "Bosch": ("bosch",),
    "Ryobi": ("ryobi",),
    "Ridgid": ("ridgid",),
    "Irwin": ("irwin",),
    "Senco": ("senco",),
    "Festool": ("festool",),
    "Kreg": ("kreg",),
    "SawStop": ("sawstop", "saw stop"),
    "Mirka": ("mirka", "hiolit", "abranet", "autonet"),
    "Freud": ("freud",),
    "Amana Tool": ("amana tool",),
    "CMT": ("cmt",),
    "Whiteside": ("whiteside",),
    "Woodpeckers": ("woodpeckers",),
    "Trex": ("trex",),
    "TimberTech": ("timbertech", "timber tech"),
    "Azek": ("azek",),
    "LP SmartSide": ("lp smartside", "smartside"),
    "James Hardie": ("james hardie", "hardieplank", "hardie"),
    "Huber": ("huber", "zip system", "advantech"),
    "CertainTeed": ("certainteed", "certain teed"),
    "Velux": ("velux",),
    "ProVia": ("provia",),
    "Philips": ("philips", "phillips"),
    "Kichler": ("kichler",),
    "Lithonia": ("lithonia",),
    "Cooper Lighting": ("cooper lighting",),
    "Satco": ("satco",),
    "Feit Electric": ("feit",),
    "Leviton": ("leviton",),
    "Square D": ("square d",),
    "Southwire": ("southwire",),
    "Hunter": ("hunter fan", "hunter"),
    "First Alert": ("first alert",),
    "BRK": ("brk",),
    "Streamlight": ("streamlight",),
    "Radians": ("radians",),
    "Marshalltown": ("marshalltown",),
    "Malco": ("malco",),
    "Hager": ("hager",),
    "Werner": ("werner",),
    "Simpson Strong-Tie": ("simpson strong-tie", "strong-tie"),
    "Frigidaire": ("frigidaire",),
    "Amana": ("amana",),
    "Prebena": ("prebena",),
    "Wera": ("wera",),
    "Vessel": ("vessel",),
    "Jet": ("jpw", "jet tools"),
    "Keystone": ("keystone",),
}

# Words that mark a supplier account as a distributor rather than a maker. These
# accounts must never reach the manufacturer line.
DISTRIBUTOR_MARKERS = (
    "supply", "supplies", "distribut", "dealers", "cooperative", "wholesale",
    "sales", "lumber", "building materials", "parksite", "warehouse",
)

# Legal and regional noise stripped when a supplier name is reused as a brand.
_LEGAL_SUFFIXES = {
    "inc", "inc.", "llc", "l.l.c.", "ltd", "ltd.", "corp", "corp.", "corporation",
    "co", "co.", "company", "companies", "plc", "gmbh", "sa", "ag", "bv", "nv",
}
_REGION_TOKENS = {"usa", "us", "u.s.", "u", "s", "america", "american", "canada",
                  "na", "n.a.", "intl", "international", "worldwide", "global"}
# Trailing category nouns: "Mirka Abrasives" is the account, "Mirka" is the brand.
_CATEGORY_NOUNS = {
    "abrasives", "abrasive", "lighting", "light", "lights", "tool", "tools",
    "wire", "cable", "machinery", "machine", "hinge", "hinges", "fan", "fans",
    "electric", "electrical", "appliance", "appliances", "gypsum", "wood",
    "energy", "solutions", "innovations", "protection", "devices", "products",
    "prod", "accessory", "accessories", "joint", "systems", "eng", "engineered",
    "mfg", "manufacturing", "industrial", "industries", "brands", "group",
    "security", "parts", "nail", "metals", "gear", "cast", "stone", "trowel",
}

_VENDOR_CODE_RE = re.compile(r"\s*\(([A-Z0-9]{3,8})\)\s*$")
_WORD_SPLIT = re.compile(r"[^a-z0-9&.+-]+")


def _clean_account(name: str) -> str:
    """Drop the trailing distributor vendor code from a supplier account name."""
    return _VENDOR_CODE_RE.sub("", str(name)).strip(" -–—")


def is_distributor_account(name: str) -> bool:
    """True when a supplier name describes a reseller rather than a maker."""
    low = _clean_account(name).lower()
    return any(marker in low for marker in DISTRIBUTOR_MARKERS)


def brand_from_account(name: str) -> str:
    """Reduce a supplier account name to the brand it trades as.

    "Mirka Abrasives Inc" -> "Mirka", "Leviton Mfg Co" -> "Leviton",
    "Velux America Inc" -> "Velux". Returns "" when nothing survives the strip.
    """
    cleaned = _clean_account(name)
    if not cleaned:
        return ""

    # "Black & Decker/dewlt" and "Woods Wire Southwire" pack two names into one
    # cell; the first is the account of record.
    cleaned = re.split(r"\s*/\s*", cleaned)[0]
    tokens = [t for t in cleaned.split() if t]

    while tokens and tokens[-1].lower().strip(".,") in _LEGAL_SUFFIXES:
        tokens.pop()
    while tokens and tokens[-1].lower().strip(".,") in _REGION_TOKENS:
        tokens.pop()
    # Only trim category nouns while a name survives them.
    while len(tokens) > 1 and tokens[-1].lower().strip(".,") in _CATEGORY_NOUNS:
        tokens.pop()

    return " ".join(tokens).strip(" -&")


@dataclass
class BrandVocabulary:
    """Brand tokens this catalogue can recognise inside free text.

    Seeded with the trade brands above, then extended from the file itself: the
    brand columns and the manufacturer-side supplier accounts of the batch being
    processed. A catalogue from a different distributor therefore recognises its
    own brands without a code change.
    """

    aliases: dict[str, str] = field(default_factory=dict)   # alias -> canonical

    @classmethod
    def build(cls, records: Iterable[ProductRecord] | None = None) -> "BrandVocabulary":
        vocab = cls()
        for canonical, alias_list in SEED_BRANDS.items():
            vocab.add(canonical, alias_list)

        for record in records or []:
            observed = record.value_of("brand")
            if observed:
                vocab.add(str(observed), ())
            account = record.value_of("supplier")
            if account and not is_distributor_account(str(account)):
                trade_name = brand_from_account(str(account))
                if trade_name:
                    vocab.add(trade_name, ())
        return vocab

    def add(self, canonical: str, aliases: Iterable[str]) -> None:
        canonical = canonical.strip()
        if not canonical:
            return
        for alias in (canonical, *aliases):
            token = alias.strip().lower()
            # One-character tokens ("s" from "U S Lumber") match everything.
            if len(token) > 1:
                self.aliases.setdefault(token, canonical)

    def find(self, text: str) -> tuple[str, str] | None:
        """Longest brand alias occurring in `text`. Returns (brand, alias hit)."""
        if not text:
            return None
        low = f" {text.lower()} "
        words = set(_WORD_SPLIT.split(low))
        best: tuple[str, str] | None = None

        for alias, canonical in self.aliases.items():
            if " " in alias or "-" in alias:
                # Multi-word aliases need a substring test, single words must not
                # match inside a longer word ("3m" inside "P3M0").
                hit = alias in low
            else:
                hit = alias in words
            if hit and (best is None or len(alias) > len(best[1])):
                best = (canonical, alias)
        return best


def _set_field(
    record: ProductRecord,
    key: str,
    value: str,
    *,
    confidence: float,
    method: Method,
    source_type: SourceType,
    source_ref: str,
    evidence: str,
    issue: ValidationIssue | None = None,
) -> None:
    """Attach a resolved identity value, preserving anything already published."""
    existing = record.get(key)
    if existing is not None and not existing.is_empty:
        return
    av = AttributeValue.make(
        key=key,
        value=value,
        source_type=source_type,
        method=method,
        source_ref=source_ref,
        evidence=evidence,
        confidence=confidence,
    )
    if issue:
        av.issues.append(issue)
    record.attributes[key] = av


def _description_text(record: ProductRecord) -> str:
    """Every piece of supplier free text a brand could be hiding in."""
    parts = [str(record.value_of("product_name", "") or ""), record.sku or ""]
    for source in record.sources:
        if source.kind == "text" and source.text:
            parts.append(source.text)
    return " ".join(p for p in parts if p)


def resolve_identity(
    record: ProductRecord, vocab: BrandVocabulary, schema: AttributeSchema | None = None
) -> list[str]:
    """Fill brand and manufacturer from the evidence already in the row.

    Returns the keys this call filled. Never overwrites a published value.
    """
    filled: list[str] = []
    account_raw = str(record.value_of("supplier", "") or "")
    account = _clean_account(account_raw)
    is_reseller = bool(account) and is_distributor_account(account)

    # ---- brand ----------------------------------------------------------
    brand = str(record.value_of("brand", "") or "")
    if not brand:
        hit = vocab.find(_description_text(record))
        if hit:
            brand, alias = hit
            _set_field(
                record, "brand", brand,
                confidence=0.88,
                method=Method.REGEX,
                source_type=SourceType.INPUT,
                source_ref="identity_resolver#description",
                evidence=f"'{alias}' appears in the supplier description",
            )
            filled.append("brand")

    if not brand and account and not is_reseller:
        trade_name = brand_from_account(account)
        if trade_name:
            brand = trade_name
            _set_field(
                record, "brand", brand,
                confidence=0.65,
                method=Method.KG_INFER,
                source_type=SourceType.INFERRED,
                source_ref="identity_resolver#supplier",
                evidence=f"Supplier account '{account}' is a manufacturer account, not a reseller",
                issue=ValidationIssue(
                    code="BRAND_FROM_SUPPLIER",
                    severity=Severity.INFO,
                    message=f"Brand taken from the manufacturer account '{account}' — no brand column or description match",
                    field_key="brand",
                    raised_by="identity_resolver",
                ),
            )
            filled.append("brand")

    # ---- manufacturer ---------------------------------------------------
    if not record.value_of("manufacturer"):
        if account and not is_reseller:
            _set_field(
                record, "manufacturer", account,
                confidence=0.80,
                method=Method.INPUT_FIELD,
                source_type=SourceType.INPUT,
                source_ref="identity_resolver#supplier",
                evidence=f"Part_Manuf: {account_raw}",
            )
            filled.append("manufacturer")
        elif brand:
            _set_field(
                record, "manufacturer", brand,
                confidence=0.70,
                method=Method.KG_INFER,
                source_type=SourceType.INFERRED,
                source_ref="identity_resolver#brand",
                evidence=(
                    f"Brand '{brand}' is the manufacturer of record; the supplied "
                    f"account '{account}' is a distributor"
                    if account else f"Brand '{brand}' is the manufacturer of record"
                ),
                issue=ValidationIssue(
                    code="MANUFACTURER_FROM_BRAND",
                    severity=Severity.INFO,
                    message=(
                        f"Supplier account '{account}' is a distributor, so the brand "
                        f"was published as the manufacturer instead"
                        if account else "Manufacturer taken from the brand"
                    ),
                    field_key="manufacturer",
                    raised_by="identity_resolver",
                ),
            )
            filled.append("manufacturer")

    if filled:
        record.log_stage("identity", filled=filled, supplier=account or None,
                         supplier_is_reseller=is_reseller)
    return filled


def resolve_batch(
    records: list[ProductRecord], schema: AttributeSchema | None = None
) -> dict[str, int]:
    """Resolve identity across a batch, using the batch to build the vocabulary."""
    vocab = BrandVocabulary.build(records)
    counts: dict[str, int] = {"brand": 0, "manufacturer": 0}
    for record in records:
        for key in resolve_identity(record, vocab, schema):
            counts[key] = counts.get(key, 0) + 1
    logger.info(
        "Identity resolved: %d brands, %d manufacturers (%d brand aliases known)",
        counts["brand"], counts["manufacturer"], len(vocab.aliases),
    )
    return counts
