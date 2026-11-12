"""Stand-in for Amazon EventBridge + the register stage: registers one weekly file every INTERVAL_S
seconds, as if a new week had been published. Re-running it registers nothing new (checksums).

BAD_FILE=1 also registers one unreadable file, to show the batch-quality gate (needs_review).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from . import db
from .stages import LAKE, register

SOURCE = Path(os.environ.get("SOURCE_DIR", "/source"))
INTERVAL_S = float(os.environ.get("INTERVAL_S", "1"))
BAD_FILE = os.environ.get("BAD_FILE", "0") == "1"


def main() -> None:
    conn = db.connect()
    files = sorted(SOURCE.glob("patents_ipa*.json"))
    if BAD_FILE:
        bad = LAKE / "incoming" / "patents_ipa250508.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text('[{"title": "TRUNCATED UPLOAD", "doc_number": "2025')  # invalid JSON
        files.insert(len(files) // 2, bad)
    print(f"[scheduler] {len(files)} weekly files to register, one every {INTERVAL_S}s", flush=True)
    for f in files:
        batch_id = register(conn, f)
        print(f"[scheduler] {f.name}: " + (f"batch {batch_id}" if batch_id else "already registered"), flush=True)
        time.sleep(INTERVAL_S)
    print("[scheduler] all files registered", flush=True)


if __name__ == "__main__":
    main()
