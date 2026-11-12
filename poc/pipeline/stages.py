"""The ingestion stages (design Section 7.1), reusing the Part 1 code in `patsearch/`.

Proof-of-concept substitutions (the stage contracts stay the same as in the design):
  * the S3 data lake is a shared Docker volume mounted at LAKE_DIR (raw/, processed/, embeddings/, index/)
  * the embed stage uses Part 1's model-free HashEmbedder (production: bge-small on GPU)
  * the index stage writes one .npz file per batch (production: bulk-index into OpenSearch)

Every stage reads its input from the lake, writes deterministic output (re-running overwrites it)
and returns statistics that are merged into batches.stats.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from patsearch.chunk import KIND_NAMES, chunk_patent
from patsearch.claims import segment_claims
from patsearch.config import ChunkConfig
from patsearch.embed import HashEmbedder
from patsearch.load import LoadStats, content_hash, dedupe, iter_patents
from patsearch.normalize import parse_cpc
from patsearch.tokens import SimpleTokenizer

from .db import NeedsReview

LAKE = Path(os.environ.get("LAKE_DIR", "/lake"))
MODEL = "hash-256"
MAX_DROP_RATE = 0.01


def _batch_dir(kind: str, batch_id: int) -> Path:
    d = LAKE / kind / str(batch_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, path)  # atomic: readers never see a half-written file


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ------------------------------------------------------------------------------------------------
# 1. register (run by the scheduler, the stand-in for EventBridge)
# ------------------------------------------------------------------------------------------------
def register(conn, source: Path) -> int | None:
    """Copy a source file into the lake and create its batch + first job. None if already registered."""
    data = source.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    m = re.search(r"ipa(\d{2})(\d{2})(\d{2})", source.name)
    week = f"20{m[1]}-{m[2]}-{m[3]}" if m else None
    raw = LAKE / "raw" / (week or "unknown") / source.name
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(data)
    with conn.transaction():
        row = conn.execute("INSERT INTO batches (source_uri, sha256, pub_week) VALUES (%s, %s, %s) "
                           "ON CONFLICT (sha256) DO NOTHING RETURNING batch_id",
                           (str(raw), sha, week)).fetchone()
        if row is None:
            return None
        conn.execute("INSERT INTO jobs (batch_id, stage) VALUES (%s, 'parse')", (row["batch_id"],))
    return row["batch_id"]


# ------------------------------------------------------------------------------------------------
# 2. parse: Part 1 loading, deduplication, claim re-assembly and chunking
# ------------------------------------------------------------------------------------------------
def parse(conn, batch: dict) -> dict:
    stats = LoadStats()
    patents = list(iter_patents([Path(batch["source_uri"])], stats))
    kept, _ = dedupe(patents)
    tok, cfg = SimpleTokenizer(), ChunkConfig()
    out_patents, out_chunks = [], []
    for p in kept:
        cpc = parse_cpc(p.classification)
        pub = str(p.pub_date) if p.pub_date else None
        out_patents.append({
            "doc_number": p.doc_number, "title": p.title, "abstract": p.abstract,
            "pub_date": f"{pub[:4]}-{pub[4:6]}-{pub[6:]}" if pub else None,
            "cpc": cpc.display if cpc else None, "content_hash": content_hash(p),
        })
        for i, c in enumerate(chunk_patent(p, segment_claims(p.claims_raw), tok, cfg)):
            out_chunks.append({"chunk_id": f"{p.doc_number}:{i}", "doc_number": p.doc_number,
                               "kind": KIND_NAMES[c.kind], "label": c.label,
                               "text": c.display_text, "embed_text": c.embed_text})
    out = _batch_dir("processed", batch["batch_id"])
    _write_jsonl(out / "patents.jsonl", out_patents)
    _write_jsonl(out / "chunks.jsonl", out_chunks)
    dropped = stats.records - stats.loaded
    result = {"records": stats.records, "patents": len(out_patents), "chunks": len(out_chunks),
              "dropped": dropped, "file_errors": len(stats.file_errors)}
    if stats.file_errors or stats.records == 0:
        raise NeedsReview("source file could not be read", result)
    if dropped / stats.records > MAX_DROP_RATE:
        raise NeedsReview(f"drop rate {dropped / stats.records:.1%} above {MAX_DROP_RATE:.0%}", result)
    return result


# ------------------------------------------------------------------------------------------------
# 3. embed
# ------------------------------------------------------------------------------------------------
def embed(conn, batch: dict) -> dict:
    chunks = _read_jsonl(_batch_dir("processed", batch["batch_id"]) / "chunks.jsonl")
    vectors = HashEmbedder(256).encode_passages([c["embed_text"] for c in chunks])
    path = LAKE / "embeddings" / MODEL / f"{batch['batch_id']}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as fh:
        np.save(fh, vectors)
    os.replace(path.with_suffix(".tmp"), path)
    norms = np.linalg.norm(vectors, axis=1) if len(vectors) else np.ones(0)
    if np.isnan(vectors).any() or (len(norms) and not np.allclose(norms[norms > 0], 1, atol=1e-4)):
        raise RuntimeError("invalid vectors (NaN or not unit length)")
    return {"vectors": int(len(vectors)), "model": MODEL}


# ------------------------------------------------------------------------------------------------
# 4. load: one transaction per batch; cross-batch dedupe on content_hash (A9 -> alias)
# ------------------------------------------------------------------------------------------------
def load(conn, batch: dict) -> dict:
    d = _batch_dir("processed", batch["batch_id"])
    patents, chunks = _read_jsonl(d / "patents.jsonl"), _read_jsonl(d / "chunks.jsonl")
    loaded, aliases = set(), []
    with conn.transaction():
        for p in patents:
            dup = conn.execute("SELECT doc_number FROM patents WHERE content_hash = %s AND doc_number <> %s",
                               (p["content_hash"], p["doc_number"])).fetchone()
            if dup:  # byte-identical republication already loaded from an earlier week
                aliases.append(f"{p['doc_number']}->{dup['doc_number']}")
                continue
            conn.execute(
                "INSERT INTO patents (doc_number, title, abstract, pub_date, cpc, content_hash, batch_id) "
                "VALUES (%(doc_number)s, %(title)s, %(abstract)s, %(pub_date)s, %(cpc)s, %(content_hash)s, %(b)s) "
                "ON CONFLICT (doc_number) DO UPDATE SET title = EXCLUDED.title, abstract = EXCLUDED.abstract, "
                "pub_date = EXCLUDED.pub_date, cpc = EXCLUDED.cpc, batch_id = EXCLUDED.batch_id, "
                "version = patents.version + (patents.content_hash IS DISTINCT FROM EXCLUDED.content_hash)::int, "
                "content_hash = EXCLUDED.content_hash, updated_at = now()",
                {**p, "b": batch["batch_id"]})
            loaded.add(p["doc_number"])
        conn.execute("DELETE FROM chunks WHERE doc_number = ANY(%s)", (list(loaded),))
        rows = [(c["chunk_id"], c["doc_number"], batch["batch_id"], c["kind"], c["label"], c["text"])
                for c in chunks if c["doc_number"] in loaded]
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO chunks (chunk_id, doc_number, batch_id, kind, label, text) "
                            "VALUES (%s, %s, %s, %s, %s, %s)", rows)
    return {"patents_loaded": len(loaded), "chunks_loaded": len(rows), "aliases": aliases}


# ------------------------------------------------------------------------------------------------
# 5. index (stand-in for the OpenSearch bulk index)
# ------------------------------------------------------------------------------------------------
def index(conn, batch: dict) -> dict:
    bid = batch["batch_id"]
    order = [c["chunk_id"] for c in _read_jsonl(_batch_dir("processed", bid) / "chunks.jsonl")]
    pos = {cid: i for i, cid in enumerate(order)}
    ids = [r["chunk_id"] for r in conn.execute(
        "SELECT chunk_id FROM chunks WHERE batch_id = %s ORDER BY chunk_id", (bid,))]
    vectors = np.load(LAKE / "embeddings" / MODEL / f"{bid}.npy")
    selected = vectors[[pos[i] for i in ids]] if ids else np.zeros((0, vectors.shape[1]), np.float32)
    path = LAKE / "index" / f"{bid}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as fh:
        np.savez(fh, ids=np.array(ids, dtype=str), vectors=selected)
    os.replace(path.with_suffix(".tmp"), path)
    return {"indexed_chunks": len(ids)}


# ------------------------------------------------------------------------------------------------
# 6. verify: counts must match, and canary queries must find their own patent
# ------------------------------------------------------------------------------------------------
def verify(conn, batch: dict) -> dict:
    bid = batch["batch_id"]
    data = np.load(LAKE / "index" / f"{bid}.npz")
    ids, vectors = list(data["ids"]), data["vectors"]
    n_db = conn.execute("SELECT count(*) AS n FROM chunks WHERE batch_id = %s", (bid,)).fetchone()["n"]
    if n_db != len(ids):
        raise NeedsReview(f"index has {len(ids)} chunks, database has {n_db}")
    canaries = [i for i, cid in enumerate(ids) if cid.endswith(":1")][:3]  # first claim chunk of a patent
    for i in canaries:
        scores = vectors @ vectors[i]
        # the chunk must be (one of) its own best matches; family patents with identical text can tie
        if scores[i] < scores.max() - 1e-6:
            raise NeedsReview(f"canary for {ids[i]} returned {ids[int(np.argmax(scores))]}")
    return {"verified_chunks": len(ids), "canaries_passed": len(canaries)}


RUNNERS = {"parse": parse, "embed": embed, "load": load, "index": index, "verify": verify}
