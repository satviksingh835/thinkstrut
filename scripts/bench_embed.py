"""Embedding micro-benchmark: precision and batch size on a fixed sample of real chunks.

    uv run python scripts/bench_embed.py --index artifacts/large_5k_baseline

Samples 10,000 chunk texts (seed 0) from the index, embeds them with float32 / batch 64 (the Part 1
setting) and with float16 at several batch sizes, and reports throughput (tokens/s) and the cosine
similarity of each float16 vector to its float32 counterpart.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patsearch.embed import SentenceTransformerEmbedder  # noqa: E402
from patsearch.normalize import title_for_embedding  # noqa: E402
from patsearch.store import resolve_artifact_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=Path("results/part3_embed_microbench.json"))
    args = ap.parse_args()
    con = sqlite3.connect(resolve_artifact_dir(args.index) / "patents.sqlite")
    n_chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    ids = np.random.default_rng(0).choice(n_chunks, size=min(args.n, n_chunks), replace=False).tolist()
    rows = con.execute(f"SELECT c.text, d.title FROM chunks c JOIN docs d USING (doc_id) "
                       f"WHERE c.chunk_id IN ({','.join(map(str, ids))})").fetchall()
    texts = [f"{title_for_embedding(t)}\n{x}" for x, t in rows]

    results, ref = [], None
    for dtype, batch in [("float32", 64), ("float16", 64), ("float16", 128), ("float16", 256)]:
        emb = SentenceTransformerEmbedder(dtype=dtype)
        n_tok = sum(emb.tokenizer.count(texts))
        emb.encode_passages(texts[:256], batch_size=batch)  # warm-up
        t = time.perf_counter()
        vecs = emb.encode_passages(texts, batch_size=batch)
        secs = time.perf_counter() - t
        if ref is None:
            ref = vecs
        cos = (vecs * ref).sum(axis=1)
        row = {"dtype": dtype, "batch_size": batch, "seconds": round(secs, 1),
               "tokens_per_s": round(n_tok / secs), "speedup_vs_fp32_b64": None,
               "cosine_vs_fp32_min": round(float(cos.min()), 5), "cosine_vs_fp32_mean": round(float(cos.mean()), 6)}
        results.append(row)
        print(json.dumps(row), flush=True)
        del emb
    for r in results:
        r["speedup_vs_fp32_b64"] = round(r["tokens_per_s"] / results[0]["tokens_per_s"], 2)
    args.out.write_text(json.dumps({"n_texts": len(texts), "results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
