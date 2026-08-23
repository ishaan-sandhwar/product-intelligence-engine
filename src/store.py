"""SQLite persistence for records, runs and reviewer decisions.

Records are stored as their full audit JSON rather than shredded into columns:
the schema is user-swappable, so a fixed column layout would fight the whole
design. The columns that do exist are the ones the dashboard filters on.

Reviewer decisions are kept in their own table even though they are also inside
the record. That table is the feedback corpus - approved corrections are fed
back as few-shot examples, which is what makes the human-in-the-loop step
improve the system instead of just patching one row.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import DB_PATH
from src.models import BatchResult, ProductRecord

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS products (
    sku            TEXT PRIMARY KEY,
    record_id      TEXT,
    category       TEXT,
    brand          TEXT,
    review_state   TEXT,
    quality_before REAL,
    quality_after  REAL,
    grade          TEXT,
    fields_filled  INTEGER,
    fields_flagged INTEGER,
    run_id         TEXT,
    payload        TEXT NOT NULL,
    updated_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_state    ON products(review_state);
CREATE INDEX IF NOT EXISTS idx_products_grade    ON products(grade);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS review_decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sku           TEXT NOT NULL,
    field_key     TEXT NOT NULL,
    decision      TEXT NOT NULL,     -- approved | edited | rejected
    machine_value TEXT,
    human_value   TEXT,
    machine_conf  REAL,
    source_type   TEXT,
    method        TEXT,
    category      TEXT,
    reviewer      TEXT,
    note          TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decisions_field ON review_decisions(field_key);
CREATE INDEX IF NOT EXISTS idx_decisions_sku   ON review_decisions(sku);

-- The products.payload column holds the *audit* view, which is lossy on
-- purpose (readable, exportable). Reviewing a record after a restart needs the
-- exact object back, so the full pydantic dump is kept beside it.
CREATE TABLE IF NOT EXISTS record_models (
    sku        TEXT PRIMARY KEY,
    model_json TEXT NOT NULL,
    updated_at TEXT
);
"""

_write_lock = threading.Lock()


@contextmanager
def connect(path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection with sensible pragmas and row access by name."""
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | str = DB_PATH) -> None:
    """Create tables if they do not exist. Safe to call repeatedly."""
    with connect(path) as conn:
        conn.executescript(SCHEMA_SQL)


# ------------------------------------------------------------------ products
def save_record(record: ProductRecord, *, run_id: str | None = None, path: Path | str = DB_PATH) -> None:
    """Upsert one record."""
    payload = json.dumps(record.to_audit_dict(), ensure_ascii=False)
    row = (
        record.sku,
        record.record_id,
        record.category,
        str(record.value_of("brand", "") or ""),
        record.review_state.value,
        record.quality_before.overall if record.quality_before else None,
        record.quality_after.overall if record.quality_after else None,
        record.quality_after.grade() if record.quality_after else None,
        len(record.filled_keys()),
        len(record.needs_review()),
        run_id,
        payload,
        record.updated_at,
    )
    with _write_lock, connect(path) as conn:
        conn.execute(
            """INSERT INTO products
               (sku, record_id, category, brand, review_state, quality_before,
                quality_after, grade, fields_filled, fields_flagged, run_id, payload, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(sku) DO UPDATE SET
                 record_id=excluded.record_id, category=excluded.category,
                 brand=excluded.brand, review_state=excluded.review_state,
                 quality_before=excluded.quality_before, quality_after=excluded.quality_after,
                 grade=excluded.grade, fields_filled=excluded.fields_filled,
                 fields_flagged=excluded.fields_flagged, run_id=excluded.run_id,
                 payload=excluded.payload, updated_at=excluded.updated_at""",
            row,
        )
        conn.execute(
            """INSERT INTO record_models (sku, model_json, updated_at) VALUES (?,?,?)
               ON CONFLICT(sku) DO UPDATE SET
                 model_json=excluded.model_json, updated_at=excluded.updated_at""",
            (record.sku, record.model_dump_json(), record.updated_at),
        )


def save_records(
    records: Iterable[ProductRecord], *, run_id: str | None = None, path: Path | str = DB_PATH
) -> int:
    """Bulk upsert. Returns the number written."""
    count = 0
    for record in records:
        save_record(record, run_id=run_id, path=path)
        count += 1
    return count


def load_record(sku: str, path: Path | str = DB_PATH) -> dict[str, Any] | None:
    """Load one record's audit dictionary."""
    with connect(path) as conn:
        row = conn.execute("SELECT payload FROM products WHERE sku=?", (sku,)).fetchone()
    return json.loads(row["payload"]) if row else None


def list_records(
    *,
    category: str | None = None,
    review_state: str | None = None,
    grade: str | None = None,
    search: str | None = None,
    limit: int = 500,
    path: Path | str = DB_PATH,
) -> list[dict[str, Any]]:
    """Summary rows for the catalogue table, without loading every payload."""
    clauses: list[str] = []
    params: list[Any] = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if review_state:
        clauses.append("review_state = ?")
        params.append(review_state)
    if grade:
        clauses.append("grade = ?")
        params.append(grade)
    if search:
        clauses.append("(sku LIKE ? OR brand LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%"])

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""SELECT sku, category, brand, review_state, quality_before,
                       quality_after, grade, fields_filled, fields_flagged, updated_at
                FROM products {where}
                ORDER BY fields_flagged DESC, quality_after ASC
                LIMIT ?"""
    params.append(limit)

    with connect(path) as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def load_all_records(path: Path | str = DB_PATH, limit: int = 2000) -> list[dict[str, Any]]:
    """Every stored record's full payload. Used for export and catalogue stats."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT payload FROM products ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def count_records(path: Path | str = DB_PATH) -> int:
    with connect(path) as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()["n"]


def clear_products(path: Path | str = DB_PATH) -> int:
    """Wipe the product table. Returns how many rows were removed."""
    with _write_lock, connect(path) as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM products").fetchone()["n"]
        conn.execute("DELETE FROM products")
        conn.execute("DELETE FROM record_models")
    return n


# ---------------------------------------------------------------------- runs
def save_run(result: BatchResult, path: Path | str = DB_PATH) -> None:
    with _write_lock, connect(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, payload, created_at) VALUES (?,?,?)",
            (result.run_id, result.model_dump_json(), result.started_at),
        )


def list_runs(limit: int = 20, path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT payload FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


# --------------------------------------------------------- review decisions
def record_decision(
    *,
    sku: str,
    field_key: str,
    decision: str,
    machine_value: Any = None,
    human_value: Any = None,
    machine_conf: float | None = None,
    source_type: str | None = None,
    method: str | None = None,
    category: str | None = None,
    reviewer: str = "reviewer",
    note: str = "",
    path: Path | str = DB_PATH,
) -> None:
    """Log one reviewer decision into the feedback corpus."""
    with _write_lock, connect(path) as conn:
        conn.execute(
            """INSERT INTO review_decisions
               (sku, field_key, decision, machine_value, human_value, machine_conf,
                source_type, method, category, reviewer, note)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sku,
                field_key,
                decision,
                json.dumps(machine_value, ensure_ascii=False, default=str),
                json.dumps(human_value, ensure_ascii=False, default=str),
                machine_conf,
                source_type,
                method,
                category,
                reviewer,
                note,
            ),
        )


def decision_stats(path: Path | str = DB_PATH) -> dict[str, Any]:
    """Aggregate reviewer agreement, used to calibrate confidence thresholds."""
    with connect(path) as conn:
        totals = {
            row["decision"]: row["n"]
            for row in conn.execute(
                "SELECT decision, COUNT(*) AS n FROM review_decisions GROUP BY decision"
            ).fetchall()
        }
        worst = [
            dict(r)
            for r in conn.execute(
                """SELECT field_key,
                          COUNT(*) AS reviewed,
                          SUM(CASE WHEN decision IN ('edited','rejected') THEN 1 ELSE 0 END) AS corrected
                   FROM review_decisions
                   GROUP BY field_key
                   HAVING reviewed >= 2
                   ORDER BY (CAST(corrected AS REAL) / reviewed) DESC, reviewed DESC
                   LIMIT 10"""
            ).fetchall()
        ]
        by_method = [
            dict(r)
            for r in conn.execute(
                """SELECT method,
                          COUNT(*) AS reviewed,
                          SUM(CASE WHEN decision = 'approved' THEN 1 ELSE 0 END) AS approved
                   FROM review_decisions
                   WHERE method IS NOT NULL
                   GROUP BY method
                   ORDER BY reviewed DESC"""
            ).fetchall()
        ]

    reviewed = sum(totals.values())
    approved = totals.get("approved", 0)
    return {
        "total_reviewed": reviewed,
        "by_decision": totals,
        "agreement_rate": round(100.0 * approved / reviewed, 1) if reviewed else 0.0,
        "most_corrected_fields": worst,
        "by_method": by_method,
    }


def feedback_examples(
    field_key: str | None = None, limit: int = 8, path: Path | str = DB_PATH
) -> list[dict[str, Any]]:
    """Past human corrections, for use as few-shot examples in later extractions."""
    query = """SELECT sku, field_key, machine_value, human_value, category, note
               FROM review_decisions
               WHERE decision = 'edited'"""
    params: list[Any] = []
    if field_key:
        query += " AND field_key = ?"
        params.append(field_key)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connect(path) as conn:
        rows = conn.execute(query, params).fetchall()

    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("machine_value", "human_value"):
            try:
                item[key] = json.loads(item[key]) if item[key] else None
            except (TypeError, json.JSONDecodeError):
                pass
        out.append(item)
    return out


# --------------------------------------------------------------- full records
def load_full(sku: str, path: Path | str = DB_PATH) -> ProductRecord | None:
    """Rebuild the exact ProductRecord for one SKU, or None if never saved."""
    with connect(path) as conn:
        row = conn.execute(
            "SELECT model_json FROM record_models WHERE sku=?", (sku,)
        ).fetchone()
    if not row:
        return None
    return ProductRecord.model_validate_json(row["model_json"])


def load_all_full(limit: int = 2000, path: Path | str = DB_PATH) -> list[ProductRecord]:
    """Rebuild every stored ProductRecord, newest first."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT model_json FROM record_models ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out: list[ProductRecord] = []
    for row in rows:
        try:
            out.append(ProductRecord.model_validate_json(row["model_json"]))
        except Exception:  # noqa: BLE001 - a schema change must not brick the catalogue
            logger.warning("Skipping unreadable stored record")
    return out
