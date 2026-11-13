"""Optimised, streaming build (Part 3). Produces the same index layout as build.py.

Differences from the Part 1 build (build.py, kept as `--profile baseline`):
  * CPU work in parallel: a process pool loads, cleans, chunks and token-counts upcoming files while
    the main process embeds the current file on the GPU (bounded look-ahead, so memory stays flat).
  * Embedding in float16 on MPS/CUDA, larger batches; duplicate doc_numbers and duplicate patents
    are dropped *before* embedding; identical chunk texts within a file are embedded once.
  * Streaming and resumable: each input file becomes a shard on disk (records + float16 vectors);
    a finalize pass assigns doc_ids in CPC order and writes SQLite and the arrays shard by shard.
  * Optional lossy setting: only the first N description chunks of each patent are indexed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pickle
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .build import _files_digest
from .chunk import KIND_ABSTRACT, KIND_CLAIM, KIND_DESCRIPTION, KIND_NAMES, chunk_patent
from .claims import segment_claims
from .config import DEFAULT_ARTIFACTS_DIR, LATEST_POINTER, MODEL_ID, MODEL_REVISION, SCHEMA_VERSION, ChunkConfig
from .embed import load_embedder
from .load import LoadStats, content_hash, iter_patents, list_data_files
from .normalize import abstract_hash, norm_title, parse_cpc
from .store import ARRAY_FILES, StoreWriter
from .timing import Stopwatch

# ------------------------------------------------------------------------------------------------
# worker side: one input file -> compact records (runs in a separate process)
# ------------------------------------------------------------------------------------------------
_TOK = None
_CFG: ChunkConfig | None = None
_MAX_DESC: int | None = None


def _init_worker(model_id: str, cfg: ChunkConfig, max_desc: int | None) -> None:
    global _TOK, _CFG, _MAX_DESC
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from .tokens import HFTokenizer, SimpleTokenizer

    _TOK = SimpleTokenizer() if model_id.startswith("hash-") else HFTokenizer.from_pretrained(model_id, MODEL_REVISION)
    _CFG, _MAX_DESC = cfg, max_desc


def prepare_file(path: str) -> dict:
    stats = LoadStats()
    records = []
    for p in iter_patents([Path(path)], stats):
        cpc = parse_cpc(p.classification)
        cs = segment_claims(p.claims_raw)
        chunks = chunk_patent(p, cs, _TOK, _CFG)
        dropped_desc = 0
        if _MAX_DESC is not None:  # lossy: keep only the first N description chunks
            seen = 0
            kept = []
            for c in chunks:
                if c.kind == KIND_DESCRIPTION:
                    seen += 1
                    if seen > _MAX_DESC:
                        dropped_desc += 1
                        continue
                kept.append(c)
            chunks = kept
        if not chunks:
            continue
        tokens = _TOK.count([c.embed_text for c in chunks])
        records.append({
            "doc": {
                "doc_number": p.doc_number, "kind": p.kind, "title": p.title, "title_norm": norm_title(p.title),
                "abstract": p.abstract, "pub_date": p.pub_date, "cpc_raw": p.classification or None,
                "cpc_subclass": cpc.subclass if cpc else None, "cpc_group": cpc.group if cpc else None,
                "cpc_compact": cpc.compact if cpc else None, "cpc_display": cpc.display if cpc else None,
                "cpc_ambiguous": int(bool(cpc and cpc.ambiguous)), "has_description": int(p.has_description),
                "n_claim_units": len(cs.units), "n_canceled": cs.n_canceled,
                "abstract_hash": abstract_hash(p.abstract), "flags": json.dumps(p.flags),
                "filename": p.filename, "source_file": p.source_file, "bibtex": p.bibtex,
            },
            "content_hash": content_hash(p),
            "claims": [{"lo": u.lo, "hi": u.hi, "label": u.label, "exact": int(u.exact),
                        "depends_on": u.depends_on, "text": u.text} for u in cs.units],
            "chunks": [{"kind": c.kind, "label": c.label, "claim_lo": c.claim_lo, "claim_hi": c.claim_hi,
                        "para_lo": c.para_lo, "para_hi": c.para_hi, "is_independent": int(c.is_independent),
                        "text": c.display_text, "embed_text": c.embed_text} for c in chunks],
            "tokens": tokens,
            "dropped_desc_chunks": dropped_desc,
        })
    return {"file": path, "stats": stats, "records": records}


# ------------------------------------------------------------------------------------------------
# main process
# ------------------------------------------------------------------------------------------------
def _merge_stats(total: LoadStats, part: LoadStats) -> None:
    for f in ("files", "records", "loaded", "dropped_not_a_dict", "dropped_no_doc_number", "dropped_no_text"):
        setattr(total, f, getattr(total, f) + getattr(part, f))
    total.file_errors += part.file_errors
    total.flags.update(part.flags)


def _ordered_results(pool: ProcessPoolExecutor, files: list[Path], lookahead: int):
    """Yield prepare_file results in file order, keeping at most `lookahead` files in flight."""
    pending, it = [], iter(files)
    for f in it:
        pending.append(pool.submit(prepare_file, str(f)))
        if len(pending) >= lookahead:
            break
    while pending:
        res = pending.pop(0).result()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.submit(prepare_file, str(nxt)))
        yield res


def fast_build(
    data_dir: Path,
    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR,
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: str = "float16",
    batch_size: int = 64,
    workers: int = 6,
    max_desc_chunks: int | None = None,
    cfg: ChunkConfig = ChunkConfig(),
    limit_files: int | None = None,
    embedder=None,
    log=print,
) -> tuple[Path, dict]:
    sw = Stopwatch()
    files = list_data_files(Path(data_dir))[: limit_files or None]
    if not files:
        raise FileNotFoundError(f"no JSON files in {data_dir}")
    with sw.stage("model_load"):
        embedder = embedder or load_embedder(model_id, device=device, dtype=dtype)
    data_digest = _files_digest(files)
    build_id = hashlib.sha1(json.dumps(
        [SCHEMA_VERSION, data_digest, embedder.model_id, embedder.revision, cfg.to_dict(), max_desc_chunks,
         getattr(embedder, "dtype", "float32")]).encode()).hexdigest()[:12]
    artifacts_root = Path(artifacts_root)
    work = artifacts_root / f"{build_id}.tmp"
    shards = work / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    # ---- phase 1: prepare (parallel CPU) + embed (GPU), one shard per input file --------------
    stats = LoadStats()
    seen_docs: set[str] = set()
    canonical_by_hash: dict[str, str] = {}
    aliases: list[tuple[str, str]] = []
    n_chunks = tokens_total = dedup_saved = dropped_desc = truncated = resumed = 0
    embed_ms = 0.0
    with sw.stage("prepare+embed (wall)"), ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(model_id, cfg, max_desc_chunks)
    ) as pool:
        for i, res in enumerate(_ordered_results(pool, files, lookahead=2)):  # GPU-bound: more look-ahead only costs RAM
            _merge_stats(stats, res["stats"])
            shard = shards / f"{i:05d}"
            keep = []
            for rec in res["records"]:
                dn, h = rec["doc"]["doc_number"], rec["content_hash"]
                if dn in seen_docs:
                    stats.duplicate_doc_numbers += 1
                    continue
                seen_docs.add(dn)
                if h in canonical_by_hash:  # byte-identical republication -> alias, not re-embedded
                    aliases.append((dn, canonical_by_hash[h]))
                    continue
                canonical_by_hash[h] = dn
                keep.append(rec)
            if shard.with_suffix(".npy").exists():  # resumed run: this file is already embedded
                with open(shard.with_suffix(".pkl"), "rb") as fh:
                    done = pickle.load(fh)
                n_chunks += sum(len(r["chunks"]) for r in done)
                tokens_total += sum(sum(r["tokens"]) for r in done)
                dropped_desc += sum(r["dropped_desc_chunks"] for r in done)
                resumed += 1
                continue
            texts = [c["embed_text"] for rec in keep for c in rec["chunks"]]
            uniq: dict[str, int] = {}
            index = [uniq.setdefault(t, len(uniq)) for t in texts]
            t_embed = time.perf_counter()
            vecs = embedder.encode_passages(list(uniq), batch_size=batch_size)
            embed_ms += (time.perf_counter() - t_embed) * 1000  # nested in the wall-clock stage
            vecs = vecs[index].astype(np.float16)
            _release_gpu_cache()  # MPS caches buffers for every input shape; free them between files
            for rec in keep:
                for c in rec["chunks"]:
                    del c["embed_text"]
            with open(shard.with_suffix(".pkl"), "wb") as fh:
                pickle.dump(keep, fh, protocol=pickle.HIGHEST_PROTOCOL)
            np.save(shard.with_suffix(".npy.tmp.npy"), vecs)
            os.replace(shard.with_suffix(".npy.tmp.npy"), shard.with_suffix(".npy"))
            n_chunks += len(texts)
            dedup_saved += len(texts) - len(uniq)
            tokens_total += sum(sum(r["tokens"]) for r in keep)
            truncated += sum(1 for r in keep for t in r["tokens"] if t + 2 > cfg.model_max_tokens)
            dropped_desc += sum(r["dropped_desc_chunks"] for r in keep)
            log(f"file {i + 1}/{len(files)}: {len(keep)} patents, {len(texts)} chunks "
                f"({len(texts) - len(uniq)} duplicate texts reused)")

    # ---- phase 2: finalize (doc_ids in CPC order; SQLite + arrays, one shard at a time) --------
    with sw.stage("finalize"):
        metas = []  # (sort key, shard, position, n_chunks)
        shard_files = sorted(shards.glob("*.pkl"))
        for s, path in enumerate(shard_files):
            with open(path, "rb") as fh:
                for pos, rec in enumerate(pickle.load(fh)):
                    d = rec["doc"]
                    metas.append(((d["cpc_subclass"] or "~", d["pub_date"] or 0, d["doc_number"]), s, pos,
                                  len(rec["chunks"])))
        metas.sort(key=lambda m: m[0])
        doc_of, lo_of, total = {}, {}, 0
        for doc_id, (_, s, pos, n) in enumerate(metas):
            doc_of[(s, pos)], lo_of[(s, pos)] = doc_id, total
            total += n
        n_docs = len(metas)
        dim = int(embedder.dim)
        emb = np.lib.format.open_memmap(work / ARRAY_FILES["emb"], mode="w+", dtype=np.float32, shape=(total, dim))
        docvec = np.zeros((n_docs, dim), dtype=np.float32)
        chunk_doc = np.zeros(total, dtype=np.int32)
        chunk_kind = np.zeros(total, dtype=np.int8)
        doc_lo = np.zeros(n_docs, dtype=np.int64)
        doc_hi = np.zeros(n_docs, dtype=np.int64)
        writer = StoreWriter(work / "patents.sqlite")
        id_by_number = {}
        unit_kinds: Counter = Counter()
        for s, path in enumerate(shard_files):
            with open(path, "rb") as fh:
                recs = pickle.load(fh)
            vecs = np.load(path.with_suffix(".npy"))
            row = 0
            doc_rows, claim_rows, chunk_rows = [], [], []
            for pos, rec in enumerate(recs):
                d_id, lo = doc_of[(s, pos)], lo_of[(s, pos)]
                n = len(rec["chunks"])
                hi = lo + n
                block = vecs[row:row + n].astype(np.float32)
                row += n
                emb[lo:hi] = block
                kinds = np.asarray([c["kind"] for c in rec["chunks"]], dtype=np.int8)
                chunk_kind[lo:hi], chunk_doc[lo:hi] = kinds, d_id
                doc_lo[d_id], doc_hi[d_id] = lo, hi
                sel = block[(kinds == KIND_ABSTRACT) | (kinds == KIND_CLAIM)]
                v = (sel if len(sel) else block).mean(axis=0)
                docvec[d_id] = v / max(float(np.linalg.norm(v)), 1e-12)
                id_by_number[rec["doc"]["doc_number"]] = d_id
                doc_rows.append({**rec["doc"], "doc_id": d_id, "chunk_lo": lo, "chunk_hi": hi})
                by_lo = {}  # shards written before the claims.py fix may repeat a claim number: merge them
                for u in rec["claims"]:
                    if u["lo"] in by_lo:
                        by_lo[u["lo"]]["text"] += " " + u["text"]
                        continue
                    unit_kinds["exact" if u["exact"] else ("open" if u["hi"] is None else "block")] += 1
                    by_lo[u["lo"]] = {**u, "doc_id": d_id}
                claim_rows.extend(by_lo.values())
                for j, c in enumerate(rec["chunks"]):
                    chunk_rows.append({**c, "chunk_id": lo + j, "doc_id": d_id})
            writer.write_docs(doc_rows)
            writer.write_claims(claim_rows)
            writer.write_chunks(chunk_rows)
        writer.write_aliases((a, id_by_number[c]) for a, c in aliases if c in id_by_number)
        writer.finalize()
        emb.flush()
        del emb
        np.save(work / ARRAY_FILES["docvec"], docvec)
        np.save(work / ARRAY_FILES["chunk_doc"], chunk_doc)
        np.save(work / ARRAY_FILES["chunk_kind"], chunk_kind)
        np.save(work / ARRAY_FILES["doc_chunk_lo"], doc_lo)
        np.save(work / ARRAY_FILES["doc_chunk_hi"], doc_hi)
        shutil.rmtree(shards)

    timings = sw.as_dict()
    timings["of which embed (GPU)"] = round(embed_ms, 3)
    embed_s = max(embed_ms / 1000, 1e-9)
    total_s = max(sw.total_ms / 1000, 1e-9)
    manifest = {
        "build_id": build_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "profile": "fast",
        "data": {"dir": str(data_dir), "files": len(files), "digest": data_digest},
        "model": {"id": embedder.model_id, "revision": embedder.revision, "dim": dim,
                  "device": getattr(embedder, "device", None), "dtype": getattr(embedder, "dtype", "float32"),
                  "batch_size": batch_size},
        "chunking": {**cfg.to_dict(), "max_desc_chunks": max_desc_chunks},
        "workers": workers,
        "resumed_shards": resumed,  # >0: embedding timings cover only the files embedded in this run
        "counts": {
            "load": stats.to_dict(),
            "docs": n_docs,
            "aliases": len(aliases),
            "docs_without_description": None,  # filled from SQLite below
            "cpc_subclasses": None,
            "claim_units": dict(unit_kinds),
            "chunks": total,
            "chunks_by_kind": {KIND_NAMES[k]: int(v) for k, v in sorted(Counter(chunk_kind.tolist()).items())},
            "duplicate_texts_reused": dedup_saved,
            "description_chunks_dropped": dropped_desc,
            "embed_tokens_total": tokens_total,
            "chunks_truncated": truncated,
        },
        "timings_ms": timings,
        "throughput": {
            "docs_per_s_end_to_end": round(n_docs / total_s, 2),
            "chunks_per_s_embed": round(n_chunks / embed_s, 1),
            "tokens_per_s_embed": round(tokens_total / embed_s, 1),
        },
    }
    with sqlite3_connect(work / "patents.sqlite") as con:
        c = manifest["counts"]
        c["docs_without_description"] = con.execute("SELECT count(*) FROM docs WHERE has_description = 0").fetchone()[0]
        c["cpc_subclasses"] = dict(con.execute(
            "SELECT coalesce(cpc_subclass, 'none'), count(*) FROM docs GROUP BY 1 ORDER BY 2 DESC LIMIT 20").fetchall())
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    out_dir = artifacts_root / build_id
    shutil.rmtree(out_dir, ignore_errors=True)
    work.rename(out_dir)
    (artifacts_root / LATEST_POINTER).write_text(build_id)
    return out_dir, manifest


def _release_gpu_cache() -> None:
    import sys

    torch = sys.modules.get("torch")  # never import torch here: see the OpenMP note in patsearch/__init__.py
    if torch is None:
        return
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort; never fail a build over cache housekeeping
        pass


def sqlite3_connect(path: Path):
    import sqlite3
    from contextlib import closing

    return closing(sqlite3.connect(path))
