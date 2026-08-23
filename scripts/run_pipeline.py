"""Headless batch runner — the same pipeline the UI drives, without Streamlit.

This is the path that matters for scale: point it at a catalogue file, get a
scored, validated, exportable catalogue plus a run summary. Safe to run in CI or
over a cron for a nightly re-score.

Usage:
    python scripts/run_pipeline.py data/raw/sample_catalog.csv
    python scripts/run_pipeline.py catalog.xlsx --limit 100 --concurrency 8
    python scripts/run_pipeline.py catalog.csv --no-llm --export outputs/exports/out.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import EXPORT_DIR, LLM_CONCURRENCY, LOG_DIR  # noqa: E402
from src import store  # noqa: E402
from src.export.delivery import (  # noqa: E402
    coverage_report,
    to_delivery_frame,
    write_delivery_csv,
)
from src.ingest.loader import attach_assets, load_catalog  # noqa: E402
from src.pipeline import PipelineOptions, run_batch  # noqa: E402
from src.schema import load_schema  # noqa: E402
from src.score import quality  # noqa: E402

logger = logging.getLogger("run_pipeline")


def configure_logging(verbose: bool) -> None:
    """Console + rotating-per-run file log, so a batch is debuggable after the fact."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")],
    )
    logger.info("Logging to %s", log_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("catalog", type=Path, help="CSV or XLSX input file")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N rows")
    parser.add_argument("--assets", type=Path, default=None,
                        help="folder of PDFs/images matched to SKUs by filename")
    parser.add_argument("--concurrency", type=int, default=LLM_CONCURRENCY)
    parser.add_argument("--provider", default=None,
                        help="force a provider (anthropic|gemini|openai|groq)")
    parser.add_argument("--no-llm", action="store_true",
                        help="deterministic only: regex, units, rules, peers")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--no-content", action="store_true")
    parser.add_argument("--no-peers", action="store_true")
    parser.add_argument("--export", type=Path, default=None,
                        help="write the flat CSV here (default: outputs/exports/)")
    parser.add_argument("--audit", type=Path, default=None,
                        help="also write the full audit JSON here")
    parser.add_argument("--delivery", type=Path, default=None,
                        help="write the Unilog delivery-template CSV here "
                             "(default: outputs/exports/delivery_<stamp>.csv)")
    parser.add_argument("--no-delivery", action="store_true",
                        help="skip the delivery-template export")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if not args.catalog.exists():
        logger.error("No such file: %s", args.catalog)
        return 1

    store.init_db()
    schema = load_schema()

    records = load_catalog(args.catalog, limit=args.limit, schema=schema)
    logger.info("Loaded %d records from %s", len(records), args.catalog)
    if not records:
        logger.error("Nothing to process.")
        return 1

    if args.assets:
        logger.info("Attached %d asset(s)", attach_assets(records, args.assets))

    options = PipelineOptions(
        use_llm=not args.no_llm,
        use_vision=not args.no_vision,
        use_judge=not args.no_judge,
        write_content=not args.no_content,
        use_peers=not args.no_peers,
        concurrency=args.concurrency,
        provider=args.provider,
    )

    last_pct = -1

    def progress(step: int, total: int, label: str) -> None:
        """Log every 10% instead of every record — batch logs stay readable."""
        nonlocal last_pct
        pct = int(100 * step / max(total, 1))
        if pct >= last_pct + 10:
            last_pct = pct
            logger.info("  %3d%%  %s", pct, label)

    started = time.time()
    processed, result = run_batch(records, options=options, schema=schema, progress=progress)
    elapsed = time.time() - started

    store.save_records(processed, run_id=result.run_id)
    store.save_run(result)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    export_path = args.export or (EXPORT_DIR / f"catalog_{stamp}.csv")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_flat_dict() for r in processed]).to_csv(export_path, index=False)

    delivery_path = None
    if not args.no_delivery:
        delivery_path = write_delivery_csv(
            processed, schema, args.delivery or (EXPORT_DIR / f"delivery_{stamp}.csv")
        )

    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(
            json.dumps([r.to_audit_dict() for r in processed], indent=2,
                       ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    stats = quality.score_catalog(processed)
    print("\n" + "=" * 62)
    print(f"run {result.run_id}   {result.succeeded}/{result.total} ok "
          f"({result.failed} failed)   {elapsed:.1f}s")
    print(f"quality      {result.avg_quality_before:5.1f}  ->  {result.avg_quality_after:5.1f}"
          f"   ({result.uplift:+.1f})")
    print(f"dimensions   completeness {stats['avg_completeness']:.0f} · "
          f"accuracy {stats['avg_accuracy']:.0f} · consistency {stats['avg_consistency']:.0f} · "
          f"richness {stats['avg_richness']:.0f}")
    print(f"grades       {stats['grades']}")
    print(f"fields       {result.fields_filled} filled · {result.fields_flagged} flagged "
          f"({stats['auto_approved_pct']:.0f}% auto-approved)")
    print(f"llm          {result.llm_calls} calls · {result.cache_hits} cache hits "
          f"· {elapsed / max(len(processed), 1):.2f}s per product")
    print(f"export       {export_path}")
    if delivery_path:
        coverage = coverage_report(to_delivery_frame(processed, schema))
        print(f"delivery     {delivery_path}")
        print(f"             {coverage['rows']} rows x {coverage['columns']} template columns · "
              f"identity {coverage['identity_pct']:.0f}% · taxonomy {coverage['taxonomy_pct']:.0f}% · "
              f"content {coverage['content_pct']:.0f}%")
        print(f"             {coverage['attribute_slots_used_avg']} attribute slots per product "
              f"(max {coverage['attribute_slots_used_max']}), "
              f"{coverage['feature_slots_used_avg']} feature slots")
    if result.errors:
        print(f"errors       {len(result.errors)} — see the log")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
