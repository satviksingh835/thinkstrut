"""Postgres job queue (design Sections 6.2 and 10).

A job is one (batch, stage) pair. Workers claim jobs with FOR UPDATE SKIP LOCKED, so any number of
workers can poll the same table without ever taking the same job. On success the next stage's job
is enqueued in the same transaction; on failure the job is retried with exponential backoff, up to
MAX_ATTEMPTS. Jobs whose worker died (no heartbeat for STALE_AFTER_S) are re-queued by the reaper.
"""

from __future__ import annotations

import json
import os
import threading
import time

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("DATABASE_URL", "postgresql://patsearch:patsearch@localhost:5433/patsearch")
STAGES = ["parse", "embed", "load", "index", "verify"]
NEXT_STAGE = dict(zip(STAGES, STAGES[1:] + [None]))
DONE_STATE = {"parse": "parsed", "embed": "embedded", "load": "loaded", "index": "indexed", "verify": "verified"}
MAX_ATTEMPTS = 3
STALE_AFTER_S = int(os.environ.get("STALE_AFTER_S", "20"))


class NeedsReview(Exception):
    """A data-quality problem: retrying will not help, so the batch is parked for a human."""

    def __init__(self, reason: str, stats: dict | None = None) -> None:
        super().__init__(reason)
        self.reason, self.stats = reason, stats or {}


def connect() -> psycopg.Connection:
    for _ in range(60):  # the database container may still be starting
        try:
            return psycopg.connect(DSN, autocommit=True, row_factory=dict_row)
        except psycopg.OperationalError:
            time.sleep(1)
    raise RuntimeError(f"database not reachable at {DSN}")


CLAIM_SQL = """
UPDATE jobs SET status = 'running', attempts = attempts + 1, worker = %(worker)s,
                heartbeat_at = now(), started_at = now(), finished_at = NULL
WHERE job_id = (SELECT job_id FROM jobs
                WHERE status = 'queued' AND not_before <= now()
                ORDER BY job_id
                FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING job_id, batch_id, stage, attempts
"""


def claim(conn, worker: str) -> dict | None:
    return conn.execute(CLAIM_SQL, {"worker": worker}).fetchone()


def complete(conn, job: dict, stats: dict | None) -> None:
    """Mark the job done, advance the batch state and enqueue the next stage, atomically."""
    with conn.transaction():
        # last_error is kept after a successful retry, so the dashboard can show what was recovered
        conn.execute("UPDATE jobs SET status = 'done', finished_at = now() WHERE job_id = %s", (job["job_id"],))
        conn.execute("UPDATE batches SET state = %s, stats = stats || %s::jsonb, updated_at = now() "
                     "WHERE batch_id = %s", (DONE_STATE[job["stage"]], json.dumps(stats or {}), job["batch_id"]))
        nxt = NEXT_STAGE[job["stage"]]
        if nxt:
            conn.execute("INSERT INTO jobs (batch_id, stage) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                         (job["batch_id"], nxt))


def fail(conn, job: dict, error: str) -> str:
    """Retry with exponential backoff (2, 4 s ...) or give up after MAX_ATTEMPTS. Returns the new status."""
    with conn.transaction():
        if job["attempts"] < MAX_ATTEMPTS:
            conn.execute("UPDATE jobs SET status = 'queued', last_error = %s, "
                         "not_before = now() + make_interval(secs => %s) WHERE job_id = %s",
                         (error, 2 ** job["attempts"], job["job_id"]))
            return "retry"
        conn.execute("UPDATE jobs SET status = 'failed', finished_at = now(), last_error = %s "
                     "WHERE job_id = %s", (error, job["job_id"]))
        conn.execute("UPDATE batches SET state = 'failed', updated_at = now() WHERE batch_id = %s",
                     (job["batch_id"],))
        return "failed"


def needs_review(conn, job: dict, reason: str, stats: dict | None = None) -> None:
    with conn.transaction():
        conn.execute("UPDATE jobs SET status = 'failed', finished_at = now(), last_error = %s "
                     "WHERE job_id = %s", (f"needs review: {reason}", job["job_id"]))
        conn.execute("UPDATE batches SET state = 'needs_review', stats = stats || %s::jsonb, "
                     "updated_at = now() WHERE batch_id = %s",
                     (json.dumps({**(stats or {}), "review_reason": reason}), job["batch_id"]))


def reap(conn) -> list[int]:
    """Re-queue jobs whose worker stopped sending heartbeats (crashed container, lost Spot instance)."""
    rows = conn.execute(
        "UPDATE jobs SET status = 'queued', last_error = 'worker lost (no heartbeat); re-queued' "
        "WHERE status = 'running' AND heartbeat_at < now() - make_interval(secs => %s) "
        "RETURNING job_id", (STALE_AFTER_S,)).fetchall()
    return [r["job_id"] for r in rows]


class Heartbeat:
    """Background thread that refreshes heartbeat_at while a job runs (own connection)."""

    def __init__(self, job_id: int, every_s: float = 3.0) -> None:
        self.job_id, self.every_s = job_id, every_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        with psycopg.connect(DSN, autocommit=True) as conn:
            while not self._stop.wait(self.every_s):
                conn.execute("UPDATE jobs SET heartbeat_at = now() WHERE job_id = %s", (self.job_id,))

    def __enter__(self) -> Heartbeat:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
