"""Approximate nearest-neighbour (ANN) search over chunk vectors with FAISS HNSW (Part 3).

`AnnIndex` implements the same `VectorIndex` interface as `ExactIndex` and keeps Part 1's cost-based
logic for filters:
  * selective filters (few rows selected): exact search on those rows only — cheap and exact;
  * broad or no filters: HNSW graph search; the filter (allowed rows) is applied *inside* the graph
    search through a bitmap selector, so results are never under-filled like a post-filter.
The HNSW stage only proposes candidate patents (over-fetching chunks); the candidates are then
re-scored exactly, so reported scores equal the exact ones and the only loss is a missed candidate.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .timing import Stopwatch
from .vindex import NEG_INF, ExactIndex, _top_order, expand_ranges

ANN_FILE = "chunks.hnsw.faiss"
EXACT_MAX_ROWS = 100_000  # selections up to this many rows are searched exactly (tuned in Part 3)


def build_ann(artifact_dir: Path, m: int = 32, ef_construction: int = 80, sq8: bool = False,
              block: int = 200_000) -> Path:
    """Build and save an HNSW index over chunks.f32.npy (rows keep their ids = chunk_ids)."""
    emb = np.load(Path(artifact_dir) / "chunks.f32.npy", mmap_mode="r")
    dim = emb.shape[1]
    if sq8:  # 8-bit scalar quantisation: 4x less memory for the stored vectors
        index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_8bit, m, faiss.METRIC_INNER_PRODUCT)
        sample = np.asarray(emb[np.linspace(0, len(emb) - 1, min(len(emb), 100_000)).astype(int)])
        index.train(sample)
    else:
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    for a in range(0, len(emb), block):
        index.add(np.ascontiguousarray(emb[a:a + block], dtype=np.float32))
    path = Path(artifact_dir) / ANN_FILE
    faiss.write_index(index, str(path))
    return path


class AnnIndex:
    def __init__(self, exact: ExactIndex, index, ef_search: int = 128, exact_max_rows: int = EXACT_MAX_ROWS):
        self.exact, self.index = exact, index
        self.ef_search, self.exact_max_rows = ef_search, exact_max_rows
        # expose what callers use from ExactIndex
        self.n_docs, self.n_rows = exact.n_docs, exact.n_rows
        self.doc_lo, self.doc_hi, self.lengths = exact.doc_lo, exact.doc_hi, exact.lengths
        self.chunk_kind, self.all_doc_ids = exact.chunk_kind, exact.all_doc_ids
        self.chunk_doc = np.repeat(np.arange(exact.n_docs, dtype=np.int64), exact.lengths)

    @classmethod
    def load(cls, exact: ExactIndex, artifact_dir: Path, **kw) -> AnnIndex | None:
        path = Path(artifact_dir) / ANN_FILE
        if not path.exists():
            return None
        return cls(exact, faiss.read_index(str(path)), **kw)

    # passthroughs ------------------------------------------------------------------------------
    def selected_rows(self, doc_ids):
        return self.exact.selected_rows(doc_ids)

    def choose_path(self, doc_ids, path="auto"):
        return self.exact.choose_path(doc_ids, path)

    def score(self, *a, **kw):
        return self.exact.score(*a, **kw)

    def doc_scores(self, *a, **kw):
        return self.exact.doc_scores(*a, **kw)

    def top_rows(self, *a, **kw):
        return self.exact.top_rows(*a, **kw)

    def doc_row_scores(self, q, doc_id):
        return self.exact.doc_row_scores(q, doc_id)

    # ANN search ------------------------------------------------------------------------------------
    def _selector(self, doc_ids, kinds, exclude):
        """Bitmap of allowed rows (None = everything allowed)."""
        if doc_ids is None and kinds is None and (exclude is None or not len(exclude)):
            return None, None
        allowed = np.ones(self.n_rows, dtype=bool) if doc_ids is None else np.zeros(self.n_rows, dtype=bool)
        if doc_ids is not None:
            allowed[expand_ranges(self.doc_lo[doc_ids], self.lengths[doc_ids])] = True
        if kinds is not None:
            allowed &= np.isin(self.chunk_kind, list(kinds))
        if exclude is not None and len(exclude):
            allowed[expand_ranges(self.doc_lo[exclude], self.lengths[exclude])] = False
        bits = np.packbits(allowed, bitorder="little")
        return faiss.IDSelectorBitmap(self.n_rows, faiss.swig_ptr(bits)), bits  # keep bits alive

    def top_docs(self, q, k, doc_ids=None, kinds=None, exclude=None, path="auto", sw=None):
        sw = sw or Stopwatch()
        if path != "auto" or (doc_ids is not None and self.selected_rows(doc_ids) <= self.exact_max_rows):
            return self.exact.top_docs(q, k, doc_ids, kinds, exclude, path, sw)
        k_chunks = max(20 * k, 200)
        with sw.stage("ann"):
            sel, _bits = self._selector(doc_ids, kinds, exclude)
            params = faiss.SearchParametersHNSW(efSearch=max(self.ef_search, k_chunks))
            if sel is not None:
                params.sel = sel
            _, rows = self.index.search(np.asarray(q, dtype=np.float32)[None], k_chunks, params=params)
        rows = rows[0][rows[0] >= 0]
        cand = np.unique(self.chunk_doc[rows])
        with sw.stage("rescore"):  # exact best-chunk score for every candidate patent
            ids, best, _ = self.exact.doc_scores(q, cand, kinds, "gather", sw)
            if exclude is not None and len(exclude):
                best = np.where(np.isin(ids, exclude), NEG_INF, best)
            order = _top_order(best, ids, k)
        return ids[order], best[order], "hnsw"
