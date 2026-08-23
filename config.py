"""Central configuration: paths, model IDs, thresholds, and scoring weights."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _adopt_streamlit_secrets() -> None:
    """Copy Streamlit Cloud secrets into the environment.

    A deployed app has no .env file — the host injects `st.secrets` instead.
    Everything downstream reads os.environ, so the two worlds are reconciled
    here rather than in every call site. Import and access are both guarded:
    the CLI runs without streamlit installed, and a local app runs without a
    secrets file.
    """
    # Touching st.secrets when no secrets file exists renders an error banner in
    # the app for every lookup, so the file is checked first. Streamlit Cloud
    # materialises its secrets into the home-directory path below.
    candidates = (
        Path.home() / ".streamlit" / "secrets.toml",
        Path(__file__).parent / ".streamlit" / "secrets.toml",
    )
    if not any(c.is_file() for c in candidates):
        return

    try:
        import streamlit as st
    except ModuleNotFoundError:
        return

    tunables = ("LLM_CONCURRENCY", "LLM_CACHE_ENABLED", "BATCH_SIZE", "GEMINI_RPM")
    keys = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY")
    for name in keys + tunables:
        if os.getenv(name):
            continue
        try:
            value = st.secrets[name]
        except Exception:      # noqa: BLE001 - no secrets file, or no such key
            continue
        if value:
            os.environ[name] = str(value)


_adopt_streamlit_secrets()

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
UPLOAD_DIR = DATA_DIR / "uploads"
SCHEMA_DIR = ROOT / "schemas"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
EXPORT_DIR = OUTPUT_DIR / "exports"
DB_PATH = DATA_DIR / "pie.db"

for _d in (RAW_DIR, PROCESSED_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ATTRIBUTE_SCHEMA_PATH = SCHEMA_DIR / "attributes.json"

# The delivery template ships as a CSV whose header row *is* the output contract.
# The exporter reads its column order from this file rather than hard-coding it,
# so a revised template needs no code change.
RESOURCES_DIR = ROOT / "resources"
DELIVERY_TEMPLATE_PATH = RESOURCES_DIR / "Unihack_ Expected Output - Delivery Format.csv"

# ---------------------------------------------------------------- llm
# Provider is chosen at runtime; first one with a usable API key wins.
PROVIDER_PRIORITY = ["anthropic", "gemini", "openai", "groq"]

MODEL_IDS = {
    "anthropic": {"text": "claude-sonnet-5", "vision": "claude-sonnet-5"},
    "gemini": {"text": "gemini-3.1-flash-lite", "vision": "gemini-3.6-flash"},
    "openai": {"text": "gpt-4o-mini", "vision": "gpt-4o"},
    "groq": {"text": "llama-3.3-70b-versatile", "vision": None},
}

API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}

LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.0          # deterministic extraction; enrichment overrides to 0.4
LLM_MAX_RETRIES = 3
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "5"))   # batch-mode semaphore
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "1") == "1"

# Requests-per-minute ceiling per provider. The client blocks on a token bucket
# instead of discovering the limit through 429s — a rejected call still burns a
# request slot, so pacing is cheaper than retrying. Gemini's free tier hard-caps
# at 15 RPM per model; 13 leaves headroom for the retry traffic.
LLM_RPM_LIMITS = {
    "anthropic": int(os.getenv("ANTHROPIC_RPM", "50")),
    "gemini": int(os.getenv("GEMINI_RPM", "13")),
    "openai": int(os.getenv("OPENAI_RPM", "60")),
    "groq": int(os.getenv("GROQ_RPM", "30")),
}
# Longest server-requested backoff we will wait out before giving up on a call.
LLM_MAX_BACKOFF_SECONDS = float(os.getenv("LLM_MAX_BACKOFF_SECONDS", "70"))

# ---------------------------------------------------------------- thresholds
# Any field below this lands in the human review queue.
REVIEW_CONFIDENCE_THRESHOLD = 0.75
# Below this we refuse to publish the field at all (kept as "candidate").
MIN_PUBLISH_CONFIDENCE = 0.40
# Fuzzy-match score (0-100) for mapping a raw attribute name to a canonical one.
ALIAS_MATCH_THRESHOLD = 84
# A near-tie between aliases pointing at different canonical keys is rejected
# rather than guessed; silent mis-mapping is worse than an unmapped column.
AMBIGUITY_MARGIN = 4.0
# Number of similar catalog products retrieved as RAG context.
RAG_TOP_K = 5

# Confidence assigned per extraction method, before evidence-based adjustment.
METHOD_BASE_CONFIDENCE = {
    "human_edit": 1.00,
    "input_field": 0.95,
    "regex": 0.92,
    "table_extract": 0.88,
    "llm_extract": 0.80,
    "vlm": 0.70,
    "web_extract": 0.72,
    "unit_convert": 0.90,   # inherits from parent, capped here
    "kg_infer": 0.55,
    "llm_generate": 0.60,
}

# Source precedence for golden-record conflict resolution (higher wins).
SOURCE_PRECEDENCE = {
    "human": 100,
    "input": 80,
    "pdf": 70,
    "web": 50,
    "image": 40,
    "catalog_sibling": 25,
    "inferred": 10,
}

# ---------------------------------------------------------------- scoring
@dataclass(frozen=True)
class QualityWeights:
    """Weights for the composite Data Quality Score (must sum to 1.0)."""

    completeness: float = 0.35
    accuracy: float = 0.25
    consistency: float = 0.20
    richness: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "richness": self.richness,
        }


QUALITY_WEIGHTS = QualityWeights()

# Content targets used by the richness metric and the content generator. The
# lengths are taken from the worked example in the delivery template, not
# invented: INVOICE_DESC really is a ~40-character uppercase line, and
# MOBILE_DESC really is a brand-first one-liner.
CONTENT_TARGETS = {
    "product_name_generic": (3, 40),    # (min_chars, max_chars)
    "mobile_desc": (40, 120),
    "invoice_desc": (18, 45),
    "short_desc": (70, 190),
    "long_desc": (180, 900),
    "retail_desc": (45, 170),
    "marketing_description": (220, 1200),
    "item_features": (5, 15),           # (min_count, max_count)
}

# How many ATTRIBUTE_LABEL/VALUE/UOM triplets the delivery template exposes is
# read from the template itself; this is only the fallback.
DELIVERY_ATTRIBUTE_SLOTS = 60
DELIVERY_FEATURE_SLOTS = 20

# ---------------------------------------------------------------- runtime
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))
RANDOM_SEED = 42
