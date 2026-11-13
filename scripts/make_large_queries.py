"""Write bench/queries_large.jsonl for the large (all-domain) corpus.

    uv run python scripts/make_large_queries.py --index artifacts/large_5k_baseline

Ten hand-written text queries across the corpus's main CPC areas, plus 10 exact independent claims
and 10 patents for the similar-patent input, sampled (seed 42) from the given index.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patsearch.store import resolve_artifact_dir  # noqa: E402

TEXT_QUERIES = [
    "training a neural network on distributed graphics processors",
    "lithium ion battery cathode material with high energy density",
    "semiconductor transistor with a gate-all-around nanosheet channel",
    "beam management and resource allocation for user equipment in a wireless network",
    "antibody that binds a tumour antigen for cancer immunotherapy",
    "surgical instrument with an articulating end effector",
    "image sensor pixel with reduced dark current noise",
    "verifying blockchain transactions for digital payments",
    "driver assistance system detecting objects with lidar",
    "organic light emitting display panel with integrated touch sensing",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("bench/queries_large.jsonl"))
    args = ap.parse_args()
    con = sqlite3.connect(resolve_artifact_dir(args.index) / "patents.sqlite")
    rng = random.Random(42)
    claims = con.execute("""SELECT d.doc_number, c.lo FROM claims c JOIN docs d USING (doc_id)
                            WHERE c.exact = 1 AND c.depends_on IS NULL AND length(c.text) > 200
                            ORDER BY d.doc_number, c.lo""").fetchall()
    docs = [r[0] for r in con.execute("SELECT doc_number FROM docs ORDER BY doc_number")]
    lines = [{"type": "text", "text": t} for t in TEXT_QUERIES]
    lines += [{"type": "claim", "ref": f"{d}:{lo}"} for d, lo in rng.sample(claims, 10)]
    lines += [{"type": "similar", "doc": d} for d in rng.sample(docs, 10)]
    args.out.write_text("".join(json.dumps(x) + "\n" for x in lines))
    print(f"wrote {len(lines)} queries to {args.out}")


if __name__ == "__main__":
    main()
