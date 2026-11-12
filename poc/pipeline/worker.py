"""Worker process: reap stale jobs, claim one job, run its stage, record the outcome; repeat.

Fault injection for the demo (environment variables):
  FAIL_RATE     probability that a job raises a transient error (simulated Spot interruption)
  CRASH_RATE    probability that the worker process dies mid-job (container restarts; the reaper
                re-queues the orphaned job once its heartbeat is STALE_AFTER_S old)
  STAGE_DELAY_S average simulated processing time per stage, so the dashboard shows progress
"""

from __future__ import annotations

import os
import random
import socket
import time

from . import db
from .stages import RUNNERS

FAIL_RATE = float(os.environ.get("FAIL_RATE", "0"))
CRASH_RATE = float(os.environ.get("CRASH_RATE", "0"))
STAGE_DELAY_S = float(os.environ.get("STAGE_DELAY_S", "0"))
POLL_S = 0.5


def log(msg: str) -> None:
    print(f"[{socket.gethostname()}] {msg}", flush=True)


def main() -> None:
    worker = socket.gethostname()
    conn = db.connect()
    log("worker started")
    while True:
        for job_id in db.reap(conn):
            log(f"re-queued job {job_id} (its worker stopped sending heartbeats)")
        job = db.claim(conn, worker)
        if job is None:
            time.sleep(POLL_S)
            continue
        batch = conn.execute("SELECT * FROM batches WHERE batch_id = %s", (job["batch_id"],)).fetchone()
        tag = f"job {job['job_id']} batch {job['batch_id']} {job['stage']} (attempt {job['attempts']})"
        with db.Heartbeat(job["job_id"]):
            try:
                if STAGE_DELAY_S:
                    time.sleep(STAGE_DELAY_S * random.uniform(0.5, 1.5))
                if random.random() < CRASH_RATE:
                    log(f"{tag}: simulated crash, worker exits")
                    os._exit(1)  # no cleanup: the job stays 'running' until the reaper finds it
                if random.random() < FAIL_RATE:
                    raise RuntimeError("simulated transient error (Spot interruption)")
                stats = RUNNERS[job["stage"]](conn, batch)
                db.complete(conn, job, stats)
                log(f"{tag}: done")
            except db.NeedsReview as exc:
                db.needs_review(conn, job, exc.reason, exc.stats)
                log(f"{tag}: needs review ({exc.reason})")
            except Exception as exc:  # noqa: BLE001 - every other error is treated as transient
                outcome = db.fail(conn, job, f"{type(exc).__name__}: {exc}")
                log(f"{tag}: {outcome} ({exc})")


if __name__ == "__main__":
    main()
