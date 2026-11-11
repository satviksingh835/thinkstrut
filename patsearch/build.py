"""Build pipeline: load → dedupe → parse (CPC, claims) → order → chunk → token check → embed →
doc vectors → SQLite + arrays → manifest. Every stage is timed (the Part 3 ingestion baseline)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .chunk import KIND_ABSTRACT, KIND_CLAIM, KIND_NAMES, Chunk, chunk_patent
from .claims import ClaimSet, segment_claims
from .config import DEFAULT_ARTIFACTS_DIR, LATEST_POINTER, SCHEMA_VERSION, ChunkConfig
from .embed import Embedder, load_embedder
from .load import LoadStats, Patent, dedupe, iter_patents, list_data_files
from .normalize import CPC, abstract_hash, norm_title, parse_cpc
from .store import ARRAY_FILES, StoreWriter
from .timing import Stopwatch


def _files_digest(files: list[Path]) -> str:
    """Hash of the input files (names + contents); feeds the build id."""
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode())
        h.update(hashlib.sha1(f.read_bytes()).digest())
    return h.hexdigest()


def _order_key(p: Patent, cpc: CPC | None) -> tuple:
    # CPC subclass first => every subclass filter is a contiguous slice of the embedding matrix
    return (cpc.subclass if cpc else "~", p.pub_date or 0, p.doc_number)


def build_index(
    data_dir: Path,
    artifacts_root: Path = DEFAULT_ARTIFACTS_DIR,
    model_id: str | None = None,
    embedder: Embedder | None = None,
    device: str | None = None,
    batch_size: int = 64,
    cfg: ChunkConfig = ChunkConfig(),
    limit_files: int | None = None,
    progress: bool = True,
    log=print,
) -> tuple[Path, dict]:
    """Run the whole indexing pipeline; returns (build directory, manifest dict).

    `embedder` can be passed in (tests use HashEmbedder); otherwise `model_id` is loaded.
    """
    sw = Stopwatch()
    files = list_data_files(Path(data_dir))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"no JSON files in {data_dir}")

    # 1. load (streamed file by file) -------------------------------------------------------
    stats = LoadStats()
    with sw.stage("load"):
        patents = list(iter_patents(files, stats))
    log(f"loaded {stats.loaded}/{stats.records} records from {stats.files} files")

    # 2. dedupe byte-identical publications ------------------------------------------------
    with sw.stage("dedupe"):
        patents, alias_map = dedupe(patents)

    # 3. parse CPC + segment claims -----------------------------------------------------------
    with sw.stage("parse"):
        parsed: list[tuple[Patent, CPC | None, ClaimSet]] = [
            (p, parse_cpc(p.classification), segment_claims(p.claims_raw)) for p in patents
        ]

    # 4. deterministic doc_id order ----------------------------------------------------------
    # A patent's doc_id is its position after this sort. Sorting by CPC subclass, then date, puts
    # e.g. all B60D patents next to each other, so "cpc=B60D" selects one contiguous block of rows.
    with sw.stage("order"):
        parsed.sort(key=lambda t: _order_key(t[0], t[1]))

    # 5. model ------------------------------------------------------------------------------------
    with sw.stage("model_load"):
        if embedder is None:
            embedder = load_embedder(model_id or "BAAI/bge-small-en-v1.5", device=device)
    tok = embedder.tokenizer

    # 6. chunk --------------------------------------------------------------------------------------
    # Chunks are appended patent by patent, so patent d owns rows [lo, hi) of the embedding matrix;
    # those two numbers (chunk_lo/chunk_hi) let the search jump straight to any patent's vectors.
    doc_rows, claim_rows, chunk_rows, alias_rows = [], [], [], []
    all_chunks: list[Chunk] = []
    chunk_doc: list[int] = []
    dropped_no_chunks = 0
    unit_kinds: Counter = Counter()
    with sw.stage("chunk"):
        for p, cpc, cs in parsed:
            chunks = chunk_patent(p, cs, tok, cfg)
            if not chunks:
                dropped_no_chunks += 1
                continue
            doc_id = len(doc_rows)
            lo = len(all_chunks)
            all_chunks.extend(chunks)
            chunk_doc.extend([doc_id] * len(chunks))
            hi = len(all_chunks)
            doc_rows.append(
                {
                    "doc_id": doc_id,
                    "doc_number": p.doc_number,
                    "kind": p.kind,
                    "title": p.title,
                    "title_norm": norm_title(p.title),
                    "abstract": p.abstract,
                    "pub_date": p.pub_date,
                    "cpc_raw": p.classification or None,
                    "cpc_subclass": cpc.subclass if cpc else None,
                    "cpc_group": cpc.group if cpc else None,
                    "cpc_compact": cpc.compact if cpc else None,
                    "cpc_display": cpc.display if cpc else None,
                    "cpc_ambiguous": int(bool(cpc and cpc.ambiguous)),
                    "has_description": int(p.has_description),
                    "n_claim_units": len(cs.units),
                    "n_canceled": cs.n_canceled,
                    "abstract_hash": abstract_hash(p.abstract),
                    "flags": json.dumps(p.flags),
                    "filename": p.filename,
                    "source_file": p.source_file,
                    "bibtex": p.bibtex,
                    "chunk_lo": lo,
                    "chunk_hi": hi,
                }
            )
            for u in cs.units:
                unit_kinds["exact" if u.exact else ("open" if u.hi is None else "block")] += 1
                claim_rows.append(
                    {
                        "doc_id": doc_id,
                        "lo": u.lo,
                        "hi": u.hi,
                        "label": u.label,
                        "exact": int(u.exact),
                        "depends_on": u.depends_on,
                        "text": u.text,
                    }
                )
            alias_rows.extend((a, doc_id) for a in p.aliases)
            for i, c in enumerate(chunks):
                chunk_rows.append(
                    {
                        "chunk_id": lo + i,
                        "doc_id": doc_id,
                        "kind": c.kind,
                        "label": c.label,
                        "claim_lo": c.claim_lo,
                        "claim_hi": c.claim_hi,
                        "para_lo": c.para_lo,
                        "para_hi": c.para_hi,
                        "is_independent": int(c.is_independent),
                        "text": c.display_text,
                    }
                )
    n_docs, n_chunks = len(doc_rows), len(all_chunks)
    log(f"{n_docs} docs -> {n_chunks} chunks")

    # 7. token check (no silent truncation) --------------------------------------------------
    # The model ignores everything after 512 tokens. Chunk sizes are chosen to stay under that;
    # this re-counts the final texts (+2 for the [CLS]/[SEP] tokens) and records any that don't.
    embed_texts = [c.embed_text for c in all_chunks]
    with sw.stage("token_check"):
        tok_counts = np.asarray(tok.count(embed_texts), dtype=np.int64)
    truncated = int(((tok_counts + 2) > cfg.model_max_tokens).sum())

    # 8. embed --------------------------------------------------------------------------------------
    # The expensive step (~90% of build time): one model forward pass per batch of chunks.
    # Result: an (n_chunks x 384) float32 matrix, row i = vector of chunk i.
    with sw.stage("embed"):
        emb = embedder.encode_passages(embed_texts, batch_size=batch_size, show_progress=progress)
    log(f"embedded {n_chunks} chunks in {sw.ms['embed'] / 1000:.1f}s")

    # 9. per-doc vectors (abstract + claims; fall back to all chunks) ------------------------
    # One summary vector per patent: the average of its abstract and claim vectors, re-normalised
    # to length 1. `similar` uses these for a fast first pass over all patents.
    chunk_kind = np.asarray([c.kind for c in all_chunks], dtype=np.int8)
    doc_lo = np.asarray([r["chunk_lo"] for r in doc_rows], dtype=np.int64)
    doc_hi = np.asarray([r["chunk_hi"] for r in doc_rows], dtype=np.int64)
    with sw.stage("docvec"):
        docvec = np.zeros((n_docs, emb.shape[1]), dtype=np.float32)
        for d in range(n_docs):
            block = emb[doc_lo[d] : doc_hi[d]]
            kinds = chunk_kind[doc_lo[d] : doc_hi[d]]
            sel = block[(kinds == KIND_ABSTRACT) | (kinds == KIND_CLAIM)]
            v = (sel if len(sel) else block).mean(axis=0)
            docvec[d] = v / max(float(np.linalg.norm(v)), 1e-12)

    # 10. write -----------------------------------------------------------------------------------
    # The build id hashes everything that determines the index (data, model, chunking, schema), so
    # rebuilding from identical inputs reproduces the same directory name. Files are written to
    # <id>.tmp and renamed at the end: a crashed build never leaves a half-written index behind.
    data_digest = _files_digest(files)
    build_id = hashlib.sha1(
        json.dumps([SCHEMA_VERSION, data_digest, embedder.model_id, embedder.revision, cfg.to_dict()]).encode()
    ).hexdigest()[:12]
    artifacts_root = Path(artifacts_root)
    out_dir = artifacts_root / build_id
    tmp_dir = artifacts_root / f"{build_id}.tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    with sw.stage("write_sqlite"):
        w = StoreWriter(tmp_dir / "patents.sqlite")
        w.write_docs(doc_rows)
        w.write_claims(claim_rows)
        w.write_chunks(chunk_rows)
        w.write_aliases(alias_rows)
        w.finalize()
    with sw.stage("write_arrays"):
        np.save(tmp_dir / ARRAY_FILES["emb"], emb)
        np.save(tmp_dir / ARRAY_FILES["docvec"], docvec)
        np.save(tmp_dir / ARRAY_FILES["chunk_doc"], np.asarray(chunk_doc, dtype=np.int32))
        np.save(tmp_dir / ARRAY_FILES["chunk_kind"], chunk_kind)
        np.save(tmp_dir / ARRAY_FILES["doc_chunk_lo"], doc_lo)
        np.save(tmp_dir / ARRAY_FILES["doc_chunk_hi"], doc_hi)

    timings = sw.as_dict()
    embed_s = max(sw.ms["embed"] / 1000, 1e-9)
    manifest = {
        "build_id": build_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data": {"dir": str(data_dir), "files": len(files), "digest": data_digest},
        "model": {
            "id": embedder.model_id,
            "revision": embedder.revision,
            "dim": int(emb.shape[1]),
            "device": getattr(embedder, "device", None),
            "batch_size": batch_size,
        },
        "chunking": cfg.to_dict(),
        "counts": {
            "load": stats.to_dict(),
            "dropped_no_chunks": dropped_no_chunks,
            "docs": n_docs,
            "aliases": len(alias_rows),
            "alias_map": alias_map,
            "docs_without_description": sum(1 for r in doc_rows if not r["has_description"]),
            "cpc_missing": sum(1 for r in doc_rows if r["cpc_subclass"] is None),
            "cpc_ambiguous_split": sum(r["cpc_ambiguous"] for r in doc_rows),
            "cpc_subclasses": dict(Counter(r["cpc_subclass"] for r in doc_rows).most_common()),
            "claim_units": dict(unit_kinds),
            "claims_canceled": sum(r["n_canceled"] for r in doc_rows),
            "chunks": n_chunks,
            "chunks_by_kind": {KIND_NAMES[k]: int(v) for k, v in sorted(Counter(chunk_kind.tolist()).items())},
            "embed_tokens_total": int(tok_counts.sum()),
            "embed_tokens_max": int(tok_counts.max()) if n_chunks else 0,
            "chunks_truncated": truncated,
        },
        "timings_ms": timings,
        "throughput": {
            "docs_per_s_end_to_end": round(n_docs / max(sw.total_ms / 1000, 1e-9), 2),
            "chunks_per_s_embed": round(n_chunks / embed_s, 1),
            "tokens_per_s_embed": round(int(tok_counts.sum()) / embed_s, 1),
        },
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    (artifacts_root / LATEST_POINTER).write_text(build_id)  # search commands open this build by default
    return out_dir, manifest
