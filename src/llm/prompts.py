"""Prompt templates for every LLM-backed stage.

Two rules run through all of them:
  1. The model must quote the exact source text it read a value from. A value
     with no quotable evidence is reported as null, never guessed.
  2. Extraction and invention are separate calls. The extractor is forbidden to
     infer; the enricher is explicitly told its output is inference and will be
     labelled as such.
"""

from __future__ import annotations

# --------------------------------------------------------------- system roles
EXTRACTOR_SYSTEM = """You are a precise industrial product data extraction engine.

Absolute rules:
- Only report a value if it is literally present in the supplied source text.
- Never infer, never estimate, never fill from general knowledge.
- For every value you report, quote the exact substring you read it from.
- If a value is absent, ambiguous, or you are unsure, return null for it.
- Preserve the source unit exactly as written; do not convert units yourself.
Returning null is always better than returning a plausible guess."""

VISION_SYSTEM = """You are an industrial product image analyst.

You read nameplates, rating plates, laser-etched part numbers, packaging labels
and datasheet diagrams. Report only text and specifications you can actually
read in the image. Quote what you see verbatim. If the image is blurry, cropped,
or the field is not visible, return null. Never guess a specification from the
general appearance of the product."""

ENRICHER_SYSTEM = """You are an industrial product data specialist producing
commerce-ready catalogue content.

You work only from the verified attributes supplied to you. You may:
- Write marketing and search copy grounded in those attributes.
- Infer an attribute that follows necessarily from engineering standards or from
  the supplied similar products, and label it as an inference with a reason.
You may not invent specifications that are not derivable from the evidence given.
Every inference must state which input it was derived from."""

JUDGE_SYSTEM = """You are a strict product data auditor.

You are given extracted product attributes with the evidence each was read from.
Your job is to find errors, not to be agreeable. Flag values that contradict the
evidence, contradict each other, are physically implausible for the product
class, or use an implausible magnitude. Approve silently when a value is sound."""

CLASSIFIER_SYSTEM = """You classify industrial products into a fixed taxonomy.
You return only a category key from the supplied list, plus a confidence and a
one-line reason. Never invent a category key that is not in the list."""


# ----------------------------------------------------------------- templates
def classify_prompt(known_text: str, categories: list[tuple[str, str]]) -> str:
    """Pick a category key for a product from its available text."""
    options = "\n".join(f"  - {key}: {label}" for key, label in categories)
    return f"""Classify this industrial product into exactly one category.

AVAILABLE CATEGORIES:
{options}

PRODUCT INFORMATION:
{known_text}

Return JSON:
{{"category": "<category_key from the list>",
  "confidence": <0.0-1.0>,
  "reason": "<one line, cite the words that decided it>"}}"""


def extract_prompt(
    source_label: str,
    source_text: str,
    attribute_block: str,
    known_values: str = "",
) -> str:
    """Extract canonical attributes from one source document."""
    known_section = (
        f"\nALREADY KNOWN (do not contradict these unless the source clearly "
        f"disagrees; if it disagrees, report the source value and say so):\n{known_values}\n"
        if known_values
        else ""
    )
    return f"""Extract product attributes from the source below.

SOURCE: {source_label}
{known_section}
ATTRIBUTES TO LOOK FOR (report the canonical key exactly as given):
{attribute_block}

SOURCE TEXT:
\"\"\"
{source_text}
\"\"\"

For each attribute you can find, return an object with:
  "value"    - the value as written in the source (keep the source unit inside
               the string if the source wrote one, e.g. "7.5 HP", "460V")
  "evidence" - the exact substring from the source text you read it from
  "certain"  - true if the source states it unambiguously, false if you inferred
               it from context or the source is ambiguous

Omit attributes you cannot find. Do not include nulls.

Return JSON:
{{"attributes": {{
    "<canonical_key>": {{"value": "...", "evidence": "...", "certain": true}}
 }},
 "unmapped": [
    {{"name": "<attribute name seen in source>", "value": "...", "evidence": "..."}}
 ]}}

Put anything specification-like that does not match a canonical key into
"unmapped" rather than discarding it."""


def vision_extract_prompt(attribute_block: str, context: str = "") -> str:
    """Read specifications off a product image (nameplate, label, packaging)."""
    context_line = f"\nKNOWN CONTEXT: {context}\n" if context else ""
    return f"""Read this industrial product image and extract what is legible.
{context_line}
Report, if visible:
1. Every line of text you can read on any nameplate, rating plate, label or
   marking, verbatim, in reading order.
2. Values for these canonical attributes:
{attribute_block}
3. What the product visually appears to be.

Return JSON:
{{"visible_text": ["<line 1>", "<line 2>"],
  "attributes": {{
     "<canonical_key>": {{"value": "...", "evidence": "<the text on the image>",
                          "certain": true, "legibility": "clear|partial|poor"}}
  }},
  "product_appearance": "<one line>",
  "image_quality": "clear|acceptable|poor"}}

If the image shows no readable specification text, return empty objects rather
than describing what you think the product might be rated at."""


def infer_prompt(
    category_label: str,
    verified_block: str,
    missing_block: str,
    similar_block: str = "",
) -> str:
    """Fill gaps by inference from verified attributes and catalogue siblings."""
    similar_section = (
        f"\nSIMILAR PRODUCTS IN THIS CATALOGUE (same category, verified data):\n"
        f"{similar_block}\n"
        if similar_block
        else ""
    )
    return f"""Infer missing attributes for this {category_label}.

VERIFIED ATTRIBUTES (extracted from sources, treat as ground truth):
{verified_block}
{similar_section}
MISSING ATTRIBUTES YOU MAY ATTEMPT:
{missing_block}

Only infer a value when it follows from the verified attributes, from an
engineering standard, or from a clear pattern in the similar products. Skip
anything that would be a guess.

For each inference give:
  "value"      - the inferred value in the canonical unit shown above
  "basis"      - "derived_from_attributes" | "engineering_standard" |
                 "catalog_pattern"
  "reasoning"  - one sentence naming the specific inputs used
  "confidence" - 0.0-1.0, your honest certainty

Return JSON:
{{"inferred": {{"<canonical_key>": {{"value": ..., "basis": "...",
                                     "reasoning": "...", "confidence": 0.0}}}},
  "skipped": ["<key you chose not to guess>"]}}"""


def content_prompt(
    category_label: str, brand: str, sku: str, attribute_block: str, targets: dict
) -> str:
    """Generate the delivery template's content set, grounded in verified attributes.

    The six description variants are not stylistic choices - each one lands in a
    different system (mobile app, invoice line, search result, product page), so
    each has its own register and length, taken from the delivery template's own
    worked example.
    """
    return f"""Write catalogue content for this product.

PRODUCT: {brand} {sku} - {category_label}

VERIFIED SPECIFICATIONS (the only facts you may state):
{attribute_block}

Write in the voice of a technical distributor: factual, specific, no hype. Do
not state any specification that is not in the list above. Do not claim
certifications, warranties, performance figures, dimensions or compatibility
that are not listed. If you do not have enough verified facts for a field,
return an empty string for it rather than inventing content.

FIELD CONTRACT (each lands in a different downstream system):

  product_name_generic  {targets['product_name_generic'][0]}-{targets['product_name_generic'][1]} chars. The generic noun for this item only -
                        "Dishwasher", "Sanding Belt", "LED Lamp". No brand, no
                        model, no adjectives.

  mobile_desc           {targets['mobile_desc'][0]}-{targets['mobile_desc'][1]} chars. Comma-separated identity line for a phone
                        screen: Manufacturer BRAND, Product Name, Series, Part Number.

  invoice_desc          {targets['invoice_desc'][0]}-{targets['invoice_desc'][1]} chars. ALL CAPS, abbreviated, no punctuation
                        beyond spaces and hyphens - it prints on an invoice line.
                        Pack the most identifying specs in, e.g.
                        "DISHWASHER LEG 5 SST 120V 15A 50-1/4IN".

  short_desc            {targets['short_desc'][0]}-{targets['short_desc'][1]} chars. Brand, series, part number, product name,
                        then the 3-5 specs a buyer picks by, comma separated.

  long_desc             {targets['long_desc'][0]}-{targets['long_desc'][1]} chars. The full specification sentence: every
                        verified attribute, comma separated, in one continuous
                        line. Specification density over readability.

  retail_desc           {targets['retail_desc'][0]}-{targets['retail_desc'][1]} chars. Shelf-edge phrasing, no brand, no part
                        number - just what it is and its headline features.

  marketing_description {targets['marketing_description'][0]}-{targets['marketing_description'][1]} chars. Two or three sentences of prose for
                        the product page. Benefit-led but every claim traceable
                        to a listed specification.

  item_features         {targets['item_features'][0]}-{targets['item_features'][1]} items. Each one a single specification phrase
                        in Title Case, e.g. "5 Wash Cycles", "Stainless Steel
                        Construction". No sentences, no repetition between items.

  with_clause           Optional. The named technology or included feature this
                        product is sold "With", e.g. "With CleanBoost". Empty
                        string if none is verified.

  includes              Optional. What ships in the box, comma separated. Empty
                        string if not verified.

  application           2-5 items. Where this product is actually used.

Return JSON with exactly these keys:
{{"product_name_generic": "...",
  "mobile_desc": "...",
  "invoice_desc": "...",
  "short_desc": "...",
  "long_desc": "...",
  "retail_desc": "...",
  "marketing_description": "...",
  "item_features": ["..."],
  "with_clause": "...",
  "includes": "...",
  "application": ["..."]}}"""


def judge_prompt(category_label: str, evidence_block: str) -> str:
    """Adversarial review pass over the assembled record."""
    return f"""Audit these extracted attributes for a {category_label}.

Each row shows: attribute = value  [source] "evidence quoted from source"

{evidence_block}

Check for:
  VALUE_CONTRADICTS_EVIDENCE - the value does not match the quoted evidence
  IMPLAUSIBLE_MAGNITUDE      - off by an order of magnitude, or impossible for this product class
  INTERNAL_CONTRADICTION     - two attributes cannot both be true
  UNIT_SUSPECT               - the number only makes sense in a different unit
  WRONG_ATTRIBUTE            - the evidence describes a different attribute than the one it was assigned to
  MISSING_CRITICAL           - a specification essential to this product class is absent

Report only real problems. Do not report a problem merely because a value is
unusual; industrial catalogues contain genuinely unusual products.

Return JSON:
{{"issues": [
    {{"field_key": "<canonical_key or null for record-level>",
      "code": "<one of the codes above>",
      "severity": "error|warning|info",
      "message": "<what is wrong, specifically>",
      "suggestion": <corrected value, or null>}}
  ],
  "verdict": "clean|minor_issues|major_issues",
  "confidence_adjustments": {{"<canonical_key>": <0.0-1.0 revised confidence>}}}}"""


def conflict_prompt(field_label: str, candidates_block: str) -> str:
    """Resolve two or more sources disagreeing on the same attribute."""
    return f"""Two or more sources disagree on "{field_label}".

CANDIDATES:
{candidates_block}

Decide which is correct. Consider: which source is more authoritative for this
kind of specification, whether the values are actually the same figure in
different units, and whether one is a transcription error.

Return JSON:
{{"winner_index": <0-based index of the correct candidate>,
  "same_value_different_units": <true|false>,
  "reasoning": "<one or two sentences>",
  "confidence": <0.0-1.0>}}"""
