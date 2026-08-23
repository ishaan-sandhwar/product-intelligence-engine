"""Ingest a catalogue (or a single sparse product) and run the pipeline over it."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import BATCH_SIZE, RAW_DIR, REVIEW_CONFIDENCE_THRESHOLD, UPLOAD_DIR
from src import store
from src.ingest.loader import attach_assets, load_catalog
from src.llm.provider import LLMClient
from src.models import (
    AttributeValue,
    Method,
    ProductRecord,
    SourceDocument,
    SourceType,
)
from src.pipeline import PipelineOptions, run_batch, run_single
from src.ui.common import bootstrap, get_schema, set_records, sidebar_status

bootstrap("Ingest & Run", "📥")
sidebar_status()

st.title("📥 Ingest & Run")
st.caption(
    "Load product data from whatever exists — a spreadsheet, a datasheet, a photo, "
    "a URL, or nothing but a part number — and let the pipeline build the golden record."
)

schema = get_schema()


# ------------------------------------------------------------------ options
def run_options(key_prefix: str) -> tuple[PipelineOptions, int | None]:
    """Render the run controls and return (options, row limit)."""
    with st.expander("Pipeline options", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_llm = st.checkbox(
                "LLM extraction", value=True, key=f"{key_prefix}_llm",
                help="Off = deterministic only: regex, unit conversion, rules, peers.",
            )
            use_vision = st.checkbox("Vision (images / nameplates)", value=True,
                                     key=f"{key_prefix}_vision")
        with col2:
            use_judge = st.checkbox(
                "Adversarial judge pass", value=True, key=f"{key_prefix}_judge",
                help="A second model pass that audits the assembled record.",
            )
            write_content = st.checkbox("Generate marketing content", value=True,
                                        key=f"{key_prefix}_content")
        with col3:
            use_peers = st.checkbox("Catalogue peers (RAG + graph)", value=True,
                                    key=f"{key_prefix}_peers")
            concurrency = st.slider("Concurrency", 1, 16, 5, key=f"{key_prefix}_conc")

        providers = LLMClient.available_providers()
        provider = st.selectbox(
            "Provider", ["auto", *providers] if providers else ["auto"],
            key=f"{key_prefix}_provider",
            help="Auto picks the first configured provider in PROVIDER_PRIORITY.",
        )
        threshold = st.slider(
            "Review threshold", 0.0, 1.0, REVIEW_CONFIDENCE_THRESHOLD, 0.05,
            key=f"{key_prefix}_threshold",
            help="Fields below this confidence go to the human review queue.",
        )
        limit = st.number_input("Row limit (0 = all)", min_value=0, value=BATCH_SIZE,
                                step=5, key=f"{key_prefix}_limit")

    st.session_state["review_threshold"] = threshold
    options = PipelineOptions(
        use_llm=use_llm,
        use_vision=use_vision,
        use_judge=use_judge,
        write_content=write_content,
        use_peers=use_peers,
        concurrency=int(concurrency),
        review_threshold=float(threshold),
        provider=None if provider == "auto" else provider,
    )
    return options, (int(limit) or None)


def execute(records: list[ProductRecord], options: PipelineOptions) -> None:
    """Run the batch with a live progress bar, then persist everything."""
    bar = st.progress(0.0, text="Starting…")
    started = time.time()

    def on_progress(step: int, total: int, label: str) -> None:
        bar.progress(min(1.0, step / max(total, 1)), text=f"{label}  ({step}/{total})")

    processed, result = run_batch(records, options=options, schema=schema,
                                  progress=on_progress)
    bar.progress(1.0, text="Done")

    store.save_records(processed, run_id=result.run_id)
    store.save_run(result)
    set_records(processed)

    st.success(
        f"Processed {result.succeeded}/{result.total} products in "
        f"{time.time() - started:.1f}s · quality {result.avg_quality_before:.0f} → "
        f"{result.avg_quality_after:.0f} ({result.uplift:+.1f})"
    )
    cols = st.columns(4)
    cols[0].metric("Fields filled", result.fields_filled)
    cols[1].metric("Flagged for review", result.fields_flagged)
    cols[2].metric("LLM calls", result.llm_calls)
    cols[3].metric("Cache hits", result.cache_hits)

    if result.errors:
        with st.expander(f"{len(result.errors)} record(s) failed"):
            st.dataframe(pd.DataFrame(result.errors), use_container_width=True,
                         hide_index=True)
    st.page_link("pages/2_Catalog.py", label="Inspect the catalogue", icon="📦")


# --------------------------------------------------------------------- tabs
tab_catalog, tab_single = st.tabs(["Catalogue file", "Single product"])

with tab_catalog:
    source = st.radio("Input source", ["Upload a file", "Pick from data/raw"],
                      horizontal=True, label_visibility="collapsed")

    path: Path | None = None
    if source == "Upload a file":
        upload = st.file_uploader("Catalogue file", type=["csv", "xlsx", "xls"])
        if upload is not None:
            path = UPLOAD_DIR / upload.name
            path.write_bytes(upload.getbuffer())
            st.caption(f"Saved to {path}")
    else:
        candidates = sorted(
            p for p in RAW_DIR.glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"}
        )
        if candidates:
            choice = st.selectbox("File", [p.name for p in candidates])
            path = RAW_DIR / choice
        else:
            st.caption("data/raw/ is empty. Run scripts/make_sample_catalog.py, "
                       "or drop the challenge dataset there.")

    assets = st.file_uploader(
        "Datasheets and images (optional) — matched to a SKU by filename",
        type=["pdf", "png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
    )

    options, limit = run_options("batch")

    if path and st.button("Load and run pipeline", type="primary", key="run_catalog"):
        records = load_catalog(path, limit=limit, schema=schema)
        st.info(f"Loaded **{len(records)}** products from {path.name}.")

        if assets:
            asset_dir = UPLOAD_DIR / "assets"
            asset_dir.mkdir(parents=True, exist_ok=True)
            for file in assets:
                (asset_dir / file.name).write_bytes(file.getbuffer())
            attached = attach_assets(records, asset_dir)
            st.info(f"Attached **{attached}** asset(s) by SKU match.")

        with st.spinner("classify → extract → resolve → rules → enrich → judge "
                        "→ route → score"):
            execute(records, options)

    if path is not None and path.exists():
        with st.expander("Preview input rows"):
            frame = (pd.read_csv(path) if path.suffix.lower() == ".csv"
                     else pd.read_excel(path))
            st.dataframe(frame.head(20), use_container_width=True)


with tab_single:
    st.markdown(
        "Minimum viable input is a part number. Everything else is optional — "
        "the engine classifies, extracts, enriches and scores from whatever it gets."
    )
    col1, col2 = st.columns(2)
    with col1:
        sku = st.text_input("SKU / part number", placeholder="1LA7 090-4AA10")
        name = st.text_input("Product name", placeholder="3-phase induction motor 1.5 kW")
        brand = st.text_input("Brand", placeholder="Siemens")
    with col2:
        url = st.text_input("Product page URL (optional)")
        free_text = st.text_area(
            "Any raw text you have (optional)", height=110,
            placeholder="Paste a spec blob, an email line, a catalogue fragment…",
        )

    docs = st.file_uploader("Datasheet or photo (optional)",
                            type=["pdf", "png", "jpg", "jpeg", "webp"],
                            accept_multiple_files=True, key="single_assets")

    single_options, _ = run_options("single")

    if st.button("Build product record", type="primary", key="run_single",
                 disabled=not sku.strip()):
        record = ProductRecord(sku=sku.strip())

        def seed(key: str, value: str) -> None:
            """Seed a user-supplied field with input-level provenance."""
            if value.strip():
                record.attributes[key] = AttributeValue.make(
                    key=key,
                    value=value.strip(),
                    source_type=SourceType.INPUT,
                    method=Method.INPUT_FIELD,
                    source_ref="manual_entry",
                )

        seed("sku", sku)
        seed("product_name", name)
        seed("brand", brand)

        if free_text.strip():
            record.sources.append(
                SourceDocument(kind="text", name="pasted_text", text=free_text.strip())
            )
        if url.strip():
            record.sources.append(
                SourceDocument(kind="web", name=url.strip(), url=url.strip())
            )
        if docs:
            asset_dir = UPLOAD_DIR / "assets"
            asset_dir.mkdir(parents=True, exist_ok=True)
            for file in docs:
                target = asset_dir / file.name
                target.write_bytes(file.getbuffer())
                kind = "pdf" if target.suffix.lower() == ".pdf" else "image"
                record.sources.append(
                    SourceDocument(kind=kind, name=file.name, path=str(target))
                )

        peers = store.load_all_full(limit=500) if single_options.use_peers else None
        with st.spinner("Building the golden record…"):
            processed = run_single(record, options=single_options, schema=schema,
                                   catalog=peers)

        store.save_record(processed)
        set_records(store.load_all_full())

        before = processed.quality_before.overall if processed.quality_before else 0.0
        after = processed.quality_after.overall if processed.quality_after else 0.0
        st.success(
            f"{processed.sku} built · quality {before:.0f} → {after:.0f} · "
            f"{len(processed.filled_keys())} fields filled · "
            f"{len(processed.needs_review())} need review"
        )
        st.page_link("pages/2_Catalog.py", label="Open in the catalogue", icon="📦")
