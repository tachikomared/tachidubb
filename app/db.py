"""SQLite-backed job store — replaces flat JSON files in jobs_db/.

Schema: one `jobs` table with job_id PK + a JSON blob column. This keeps
the existing in-memory dict API intact (all code that does `jobs[id]` still
works) while adding durable, queryable storage.

Migration: on startup, any existing jobs_db/*.json files are imported into
SQLite automatically so there is zero data loss when upgrading.

Why SQLite over flat JSON:
  - O(1) read/write per job instead of O(n) directory scan on load
  - ACID writes (no corrupt JSON on crash mid-write)
  - Fast filtered queries (list_jobs by status, batch_id, etc.)
  - Single file — easier to backup/move than hundreds of JSON blobs
"""
import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("tachidubb.db")

_DB_PATH: Optional[Path] = None
_lock = asyncio.Lock()

# Fields that are large and excluded from the DB listing query
_LARGE_FIELDS = {"transcript", "transcript_raw", "_pending_args"}


def init_db(db_path: Path) -> None:
    """Create table and migrate existing JSON files. Synchronous — call once at startup."""
    global _DB_PATH
    _DB_PATH = db_path

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            data   TEXT NOT NULL,
            status TEXT GENERATED ALWAYS AS (json_extract(data, '$.status')) VIRTUAL,
            created REAL GENERATED ALWAYS AS (json_extract(data, '$.created')) VIRTUAL,
            batch_id TEXT GENERATED ALWAYS AS (json_extract(data, '$.batch_id')) VIRTUAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_batch   ON jobs(batch_id)")
    conn.commit()

    # Migrate existing JSON files
    json_dir = db_path.parent / "jobs_db"
    if json_dir.exists():
        migrated = 0
        for jf in json_dir.glob("*.json"):
            try:
                with open(jf, encoding="utf-8") as f:
                    data = json.load(f)
                jid = data.get("id") or jf.stem
                conn.execute(
                    "INSERT OR IGNORE INTO jobs(job_id, data) VALUES(?, ?)",
                    (jid, json.dumps(data, default=str, ensure_ascii=False)),
                )
                migrated += 1
            except Exception as e:
                log.warning(f"[db] Could not migrate {jf.name}: {e}")
        if migrated:
            conn.commit()
            log.info(f"[db] Migrated {migrated} JSON jobs to SQLite")

    conn.close()
    log.info(f"[db] SQLite store ready: {db_path}")


def save_job_sync(job: dict) -> None:
    """Write one job to SQLite (synchronous). Used by the pipeline during active runs."""
    if _DB_PATH is None:
        return
    persist = _strip_large_fields(job)
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO jobs(job_id, data) VALUES(?, ?)",
            (job["id"], json.dumps(persist, default=str, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"[db] save_job_sync failed: {e}")


async def save_job_async(job: dict) -> None:
    """Write one job to SQLite (async-safe, uses lock to avoid contention)."""
    if _DB_PATH is None:
        return
    persist = _strip_large_fields(job)
    blob = json.dumps(persist, default=str, ensure_ascii=False)
    async with _lock:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_blob, job["id"], blob)


def _write_blob(job_id: str, blob: str) -> None:
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute(
            "INSERT OR REPLACE INTO jobs(job_id, data) VALUES(?, ?)",
            (job_id, blob),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"[db] _write_blob failed: {e}")


def load_all_jobs() -> dict:
    """Load all job records into an in-memory dict {job_id: dict}."""
    if _DB_PATH is None or not _DB_PATH.exists():
        return {}
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        rows = conn.execute("SELECT job_id, data FROM jobs").fetchall()
        conn.close()
        result = {}
        for jid, blob in rows:
            try:
                d = json.loads(blob)
                result[jid] = d
            except Exception as e:
                log.warning(f"[db] Could not parse job {jid}: {e}")
        log.info(f"[db] Loaded {len(result)} jobs from SQLite")
        return result
    except Exception as e:
        log.error(f"[db] load_all_jobs failed: {e}")
        return {}


def delete_job_db(job_id: str) -> None:
    """Remove a job record from SQLite."""
    if _DB_PATH is None:
        return
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"[db] delete_job failed: {e}")


def _strip_large_fields(job: dict) -> dict:
    """Exclude large transient fields before persisting."""
    out = {k: v for k, v in job.items() if k not in _LARGE_FIELDS}
    # Keep a small transcript preview (first 5 segments)
    if "transcript" in job and isinstance(job["transcript"], list):
        out["transcript"] = job["transcript"][:5]
        if len(job["transcript"]) > 5:
            out["transcript_truncated"] = True
    return out
