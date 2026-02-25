# src/modules/job_store.py
"""
JobStore: SQLite-backed persistent storage for scraped + matched jobs.

Responsibilities:
  - Create and maintain the jobs database on first run
  - Save new jobs (skip duplicates via job_id PRIMARY KEY)
  - Update application status with timestamps
  - Query jobs by status, score, date
  - Track the full application lifecycle

Schema:
    job_id          TEXT PRIMARY KEY  — 16-char dedup hash
    title           TEXT
    company         TEXT
    location        TEXT
    url             TEXT
    source          TEXT              — "linkedin", "indeed", etc.
    match_score     REAL              — raw score from MatchingEngine
    matched_skills  TEXT              — JSON list
    missing_skills  TEXT              — JSON list
    rank_label      TEXT              — "Top 25%", etc.
    status          TEXT              — see APPLICATION_STATUSES
    applied_date    TEXT              — ISO date, set when status→APPLIED
    last_updated    TEXT              — ISO datetime, updated on any change
    notes           TEXT              — free-text user notes
    found_date      TEXT              — ISO date when first scraped
    date_posted     TEXT              — ISO date from job listing
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib  import Path
from typing   import Dict, List, Optional

logger = logging.getLogger(__name__)

APPLICATION_STATUSES = {
    "NOT_APPLIED",
    "APPLIED",
    "INTERVIEWING",
    "OFFER_RECEIVED",
    "REJECTED",
    "DECLINED",
}

STATUS_LABELS = {
    "NOT_APPLIED":    "○ Not Applied",
    "APPLIED":        "✓ Applied",
    "INTERVIEWING":   "⟳ Interviewing",
    "OFFER_RECEIVED": "★ Offer!",
    "REJECTED":       "✗ Rejected",
    "DECLINED":       "✗ Declined",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    title           TEXT    NOT NULL,
    company         TEXT    NOT NULL DEFAULT '',
    location        TEXT             DEFAULT '',
    url             TEXT             DEFAULT '',
    source          TEXT             DEFAULT '',
    match_score     REAL             DEFAULT 0.0,
    matched_skills  TEXT             DEFAULT '[]',
    missing_skills  TEXT             DEFAULT '[]',
    rank_label      TEXT             DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'NOT_APPLIED',
    applied_date    TEXT             DEFAULT NULL,
    last_updated    TEXT    NOT NULL,
    notes           TEXT             DEFAULT '',
    found_date      TEXT    NOT NULL,
    date_posted     TEXT             DEFAULT ''
)
"""


class JobStore:
    """
    SQLite-backed persistent job tracker.

    Uses a single persistent connection so in-memory databases
    (:memory:) work correctly in tests — each new sqlite3.connect()
    call to :memory: creates a completely separate empty database,
    so we must reuse one connection for the lifetime of this object.

    Usage:
        store = JobStore("data/jobs.db")   # file-based (production)
        store = JobStore(":memory:")       # in-memory  (tests)

        store.save_job(scraped_job, ranking_result)
        store.update_status(job_id, "APPLIED")
        jobs  = store.get_all()
        stats = store.get_stats()
        store.close()
    """

    def __init__(self, db_path: str = "data/jobs.db"):
        self.db_path = db_path

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Single persistent connection — required for :memory: correctness
        self._conn             = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

        self._init_db()
        logger.info(f"JobStore ready: {db_path}")

    # ------------------------------------------------------------------
    # CONNECTION
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("JobStore connection closed")

    # ------------------------------------------------------------------
    # PUBLIC API — WRITE
    # ------------------------------------------------------------------

    def save_job(self, scraped_job, ranking_result) -> bool:
        """
        Save a matched job to the database.

        Silently skips duplicates (same job_id already exists).

        Returns:
            True  if the job was newly inserted
            False if it was already in the database (duplicate)
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        matched = json.dumps(
            getattr(ranking_result.match_result, "matched_skills", [])
        )
        missing = json.dumps(
            getattr(ranking_result.match_result, "missing_skills", [])
        )

        sql = """
            INSERT OR IGNORE INTO jobs (
                job_id, title, company, location, url, source,
                match_score, matched_skills, missing_skills, rank_label,
                status, last_updated, found_date, date_posted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            scraped_job.job_id,
            scraped_job.raw_title,
            scraped_job.company,
            scraped_job.location,
            scraped_job.url,
            scraped_job.source,
            round(ranking_result.overall_score, 4),
            matched,
            missing,
            ranking_result.rank_label,
            "NOT_APPLIED",
            now,
            now[:10],
            scraped_job.date_posted,
        )

        cur = self._conn.execute(sql, params)
        self._conn.commit()
        inserted = cur.rowcount > 0

        if inserted:
            logger.debug(
                f"Saved: {scraped_job.raw_title} @ {scraped_job.company} "
                f"(score={ranking_result.overall_score:.3f})"
            )
        else:
            logger.debug(f"Duplicate skipped: {scraped_job.job_id}")

        return inserted

    def update_status(self, job_id: str, status: str) -> bool:
        """
        Update the application status of a job.

        Automatically sets applied_date when status → APPLIED.
        Always updates last_updated.

        Raises:
            ValueError if status is not a valid APPLICATION_STATUS
        """
        if status not in APPLICATION_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Valid: {sorted(APPLICATION_STATUSES)}"
            )

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if status == "APPLIED":
            sql    = """
                UPDATE jobs
                SET status = ?, last_updated = ?, applied_date = ?
                WHERE job_id = ?
            """
            params = (status, now, now[:10], job_id)
        else:
            sql    = """
                UPDATE jobs
                SET status = ?, last_updated = ?
                WHERE job_id = ?
            """
            params = (status, now, job_id)

        cur     = self._conn.execute(sql, params)
        self._conn.commit()
        updated = cur.rowcount > 0

        if updated:
            logger.info(f"Status updated: {job_id} → {status}")
        else:
            logger.warning(f"update_status: job_id not found: {job_id}")

        return updated

    def add_note(self, job_id: str, note: str) -> bool:
        """Save a free-text note for a job (replaces existing note)."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = self._conn.execute(
            "UPDATE jobs SET notes = ?, last_updated = ? WHERE job_id = ?",
            (note.strip(), now, job_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_job(self, job_id: str) -> bool:
        """Permanently delete a job record."""
        cur = self._conn.execute(
            "DELETE FROM jobs WHERE job_id = ?", (job_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # PUBLIC API — READ
    # ------------------------------------------------------------------

    def get_all(
        self,
        status_filter: Optional[List[str]] = None,
        min_score:     float               = 0.0,
        order_by:      str                 = "match_score DESC",
    ) -> List[Dict]:
        """
        Fetch jobs from the database.

        Args:
            status_filter: List of statuses to include.
                           None = return all statuses.
            min_score:     Only return jobs with match_score >= value.
            order_by:      SQL ORDER BY clause.

        Returns:
            List of dicts. matched_skills and missing_skills are
            Python lists (not JSON strings).
        """
        conditions = ["match_score >= ?"]
        params     = [min_score]

        if status_filter:
            placeholders = ",".join("?" * len(status_filter))
            conditions.append(f"status IN ({placeholders})")
            params.extend(status_filter)

        where = " AND ".join(conditions)
        sql   = f"SELECT * FROM jobs WHERE {where} ORDER BY {order_by}"

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Fetch a single job by job_id. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_stats(self) -> Dict:
        """
        Return counts by status for the TUI dashboard header.

        Returns dict with keys:
            total, not_applied, applied, interviewing,
            offer_received, rejected, declined,
            best_score, mean_score
        """
        sql = """
            SELECT
                COUNT(*)                                                  AS total,
                SUM(CASE WHEN status='NOT_APPLIED'    THEN 1 ELSE 0 END) AS not_applied,
                SUM(CASE WHEN status='APPLIED'        THEN 1 ELSE 0 END) AS applied,
                SUM(CASE WHEN status='INTERVIEWING'   THEN 1 ELSE 0 END) AS interviewing,
                SUM(CASE WHEN status='OFFER_RECEIVED' THEN 1 ELSE 0 END) AS offer_received,
                SUM(CASE WHEN status='REJECTED'       THEN 1 ELSE 0 END) AS rejected,
                SUM(CASE WHEN status='DECLINED'       THEN 1 ELSE 0 END) AS declined,
                MAX(match_score)                                          AS best_score,
                AVG(match_score)                                          AS mean_score
            FROM jobs
        """
        row = self._conn.execute(sql).fetchone()

        if not row or row["total"] == 0:
            return {
                "total": 0, "not_applied": 0, "applied": 0,
                "interviewing": 0, "offer_received": 0,
                "rejected": 0, "declined": 0,
                "best_score": 0.0, "mean_score": 0.0,
            }

        return {
            "total":          row["total"]          or 0,
            "not_applied":    row["not_applied"]     or 0,
            "applied":        row["applied"]         or 0,
            "interviewing":   row["interviewing"]    or 0,
            "offer_received": row["offer_received"]  or 0,
            "rejected":       row["rejected"]        or 0,
            "declined":       row["declined"]        or 0,
            "best_score":     round(row["best_score"] or 0.0, 4),
            "mean_score":     round(row["mean_score"] or 0.0, 4),
        }

    def job_exists(self, job_id: str) -> bool:
        """Return True if a job_id already exists in the database."""
        return self._conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ? LIMIT 1", (job_id,)
        ).fetchone() is not None

    def count(self, status_filter: Optional[List[str]] = None) -> int:
        """Return count of jobs, optionally filtered by status."""
        if status_filter:
            placeholders = ",".join("?" * len(status_filter))
            sql    = f"SELECT COUNT(*) FROM jobs WHERE status IN ({placeholders})"
            params = status_filter
        else:
            sql, params = "SELECT COUNT(*) FROM jobs", []

        return self._conn.execute(sql, params).fetchone()[0]

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create jobs table if it doesn't exist."""
        self._conn.execute(CREATE_TABLE_SQL)
        self._conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert sqlite3.Row to plain dict, deserialising JSON lists."""
        d = dict(row)
        d["matched_skills"] = json.loads(d.get("matched_skills") or "[]")
        d["missing_skills"] = json.loads(d.get("missing_skills") or "[]")
        d["status_label"]   = STATUS_LABELS.get(d["status"], d["status"])
        return d
