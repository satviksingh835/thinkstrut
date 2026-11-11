"""Vector index over chunk embeddings.

The engine only talks to the `VectorIndex` protocol (top_docs / top_rows / doc_row_scores), so
Part 3 can drop in an ANN implementation (FAISS IVF/HNSW with ID selectors or per-CPC partitions).

`ExactIndex` is brute force (one matrix-vector product), with three ways to restrict scoring to a
pre-filtered set of patents — chosen by cost:
  * slices  — the selection is a few contiguous row runs (e.g. one CPC subclass, because doc_ids
              are ordered by subclass, then date): score zero-copy views of the matrix
  * gather  — scattered selection covering < 50% of rows: copy the selected rows, then score them
  * full    — scattered selection covering most rows: score every row, then keep the selected docs
Work is proportional to the number of selected rows, so restrictive filters make search faster.

Data layout this relies on (built by build.py):
    emb               (n_rows x 384) matrix, one unit-length vector per chunk
    doc_lo, doc_hi    patent d's chunks are rows doc_lo[d] .. doc_hi[d]-1 (contiguous)
    chunk_kind        per row: 0 abstract, 1 claim, 2 description, 3 title
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .timing import Stopwatch

NEG_INF = np.float32(-np.inf)  # score given to excluded rows/docs so they can never be selected
FULL_FRACTION = 0.5  # above this share of rows, scoring everything beats gathering
MAX_SLICES = 64  # up to this many contiguous runs, per-run matmuls beat a gather copy


class VectorIndex(Protocol):
    def top_docs(self, q, k, doc_ids=None, kinds=None, exclude=None, path="auto", sw=None):
        """Best k patents for query vector q: (doc_ids, scores, path used)."""
        ...

    def top_rows(self, q, k, doc_ids=None, kinds=None, exclude=None, path="auto", sw=None):
        """Best k individual chunks (passages): (row ids, scores, path used)."""
        ...

    def doc_row_scores(self, q, doc_id):
        """Scores of every chunk of one patent (used to pick evidence passages)."""
        ...


@dataclass
class Scored:
    """Row scores for a set of selected patents, grouped patent by patent.

    Example: selected docs [3, 7] with 2 and 3 chunks ->
        lengths [2, 3], offsets [0, 2], scores [s3a, s3b, s7a, s7b, s7c]
    so patent doc_ids[i] owns scores[offsets[i] : offsets[i] + lengths[i]].
    """

    doc_ids: np.ndarray  # selected docs, ascending
    offsets: np.ndarray  # start of each doc's rows within `scores`
    lengths: np.ndarray  # number of rows per doc
    scores: np.ndarray  # row scores, grouped by doc in doc_id order
    path: str


def expand_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate arange(s, s+l) for all (s, l) without a Python loop.

    starts [10, 50], lengths [3, 2] -> [10, 11, 12, 50, 51]  (the row ids of the selected patents)
    """
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    # position i of the output belongs to range j; output[i] = i + (starts[j] - outputs before range j)
    before = np.cumsum(lengths) - lengths
    return np.arange(total, dtype=np.int64) + np.repeat(starts - before, lengths)


def _top_order(scores: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k scores, ties broken by ascending id — deterministic across code paths."""
    valid = np.flatnonzero(scores > NEG_INF)  # drop excluded / filtered-out entries
    if valid.size == 0 or k <= 0:
        return valid[:0]
    s = scores[valid]
    if valid.size > k:
        # argpartition finds the k largest in O(n) without sorting everything; only those get sorted
        part = np.argpartition(-s, k - 1)[:k]
        threshold = s[part].min()
        keep = np.flatnonzero(s >= threshold)  # include every tie at the boundary, then sort
        valid, s = valid[keep], s[keep]
    order = np.lexsort((ids[valid], -s))  # sort by score descending, then id ascending
    return valid[order[:k]]


class ExactIndex:
    """Brute-force ("exact") vector search: every allowed chunk is scored against the query.

    At this corpus size that is ~0.5 ms; an approximate index (ANN) only pays off at millions of rows.
    """

    def __init__(self, emb: np.ndarray, doc_lo: np.ndarray, doc_hi: np.ndarray, chunk_kind: np.ndarray) -> None:
        self.emb = emb
        self.doc_lo = doc_lo.astype(np.int64, copy=False)
        self.doc_hi = doc_hi.astype(np.int64, copy=False)
        self.lengths = self.doc_hi - self.doc_lo
        if (self.lengths <= 0).any():
            raise ValueError("every document needs at least one chunk")
        self.chunk_kind = chunk_kind
        self.n_docs = int(doc_lo.shape[0])
        self.n_rows = int(emb.shape[0])
        self.all_doc_ids = np.arange(self.n_docs, dtype=np.int64)

    # ------------------------------------------------------------------------------------------
    def selected_rows(self, doc_ids: np.ndarray | None) -> int:
        """How many chunk vectors the selected patents have (None = all patents)."""
        return self.n_rows if doc_ids is None else int(self.lengths[doc_ids].sum())

    def choose_path(self, doc_ids: np.ndarray | None, path: str = "auto") -> str:
        """Pick the cheapest way to score the selected patents (see module docstring)."""
        if doc_ids is None:
            return "full"
        if path != "auto":
            return path
        # Few contiguous runs: zero-copy views, cost ∝ selected rows — never worse than `full`.
        n_runs = 1 + int(np.count_nonzero(np.diff(doc_ids) != 1)) if doc_ids.size else 0
        if n_runs <= MAX_SLICES:
            return "slices"
        # Scattered: gathering copies the selected rows, which only pays off for a minority of rows.
        return "full" if self.selected_rows(doc_ids) >= FULL_FRACTION * self.n_rows else "gather"

    def score(
        self,
        q: np.ndarray,
        doc_ids: np.ndarray | None = None,
        kinds: set[int] | None = None,
        path: str = "auto",
        sw: Stopwatch | None = None,
    ) -> Scored:
        """Cosine score of query vector q against every chunk of the selected patents.

        doc_ids: the pre-filter's result (sorted), or None for all patents.
        kinds:   allowed chunk kinds ("scope"); chunks of other kinds get -inf.
        """
        sw = sw or Stopwatch()
        path = self.choose_path(doc_ids, path)
        q = np.asarray(q, dtype=np.float32)
        with sw.stage("score"):
            if doc_ids is None:
                # no filter: one matrix-vector product = the dot product (= cosine, since all
                # vectors have length 1) of q with every chunk vector at once
                ids, lengths, offsets = self.all_doc_ids, self.lengths, self.doc_lo
                scores = self.emb @ q
                kind_rows = self.chunk_kind
            else:
                ids = np.asarray(doc_ids, dtype=np.int64)
                lengths = self.lengths[ids]  # chunks per selected patent
                offsets = np.cumsum(lengths) - lengths  # where each patent starts in `scores`
                if path == "full":
                    # score every row, then keep only the selected patents' rows
                    rows = expand_ranges(self.doc_lo[ids], lengths)
                    scores = (self.emb @ q)[rows]
                    kind_rows = self.chunk_kind[rows]
                elif path == "slices":
                    # Consecutive doc_ids (3, 4, 5, ...) form a "run"; because each patent's rows are
                    # contiguous, a run of patents is one contiguous block of rows. emb[a:b] is a
                    # view (no copy), so each block is scored in place.
                    if ids.size:
                        brk = np.flatnonzero(np.diff(ids) != 1) + 1  # where a new run starts
                        first = ids[np.r_[0, brk]]
                        last = ids[np.r_[brk, ids.size] - 1]
                        runs = list(zip(self.doc_lo[first].tolist(), self.doc_hi[last].tolist()))
                    else:
                        runs = []
                    parts = [self.emb[a:b] @ q for a, b in runs]
                    scores = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
                    kind_rows = (
                        np.concatenate([self.chunk_kind[a:b] for a, b in runs]) if runs else np.zeros(0, np.int8)
                    )
                elif path == "gather":
                    # scattered patents: copy their rows into a new small matrix, then score it
                    rows = expand_ranges(self.doc_lo[ids], lengths)
                    scores = np.asarray(self.emb[rows]) @ q
                    kind_rows = self.chunk_kind[rows]
                else:
                    raise ValueError(f"unknown path {path!r}")
            if kinds is not None:
                # lookup table indexed by kind code: lut[kind] is True if that kind is in scope
                lut = np.zeros(256, dtype=bool)
                lut[list(kinds)] = True
                scores = np.where(lut[kind_rows], scores, NEG_INF)
        return Scored(ids, offsets, lengths, scores, path)

    # ------------------------------------------------------------------------------------------
    def doc_scores(self, q, doc_ids=None, kinds=None, path="auto", sw=None) -> tuple[np.ndarray, np.ndarray, str]:
        """Best-chunk score per selected doc (−inf if it has no chunk of the allowed kinds)."""
        sw = sw or Stopwatch()
        s = self.score(q, doc_ids, kinds, path, sw)
        with sw.stage("aggregate"):
            # reduceat takes the max over each segment scores[offsets[i]:offsets[i+1]] in one
            # vectorised call, i.e. "a patent's score = its best chunk's score"
            best = np.maximum.reduceat(s.scores, s.offsets) if s.doc_ids.size else np.zeros(0, np.float32)
        return s.doc_ids, best, s.path

    def top_docs(self, q, k, doc_ids=None, kinds=None, exclude=None, path="auto", sw=None):
        """Top-k patents by best-chunk score. `exclude`: doc_ids never to return (e.g. the query's own patent)."""
        sw = sw or Stopwatch()
        ids, best, used = self.doc_scores(q, doc_ids, kinds, path, sw)
        with sw.stage("topk"):
            if exclude is not None and len(exclude):
                best = best.copy()
                best[np.isin(ids, exclude)] = NEG_INF
            order = _top_order(best, ids, k)
        return ids[order], best[order], used

    def top_rows(self, q, k, doc_ids=None, kinds=None, exclude=None, path="auto", sw=None):
        """Top-k individual chunks ("passage" granularity: several may come from one patent)."""
        sw = sw or Stopwatch()
        s = self.score(q, doc_ids, kinds, path, sw)
        with sw.stage("topk"):
            rows = expand_ranges(self.doc_lo[s.doc_ids], s.lengths)  # row id of each score
            scores = s.scores
            if exclude is not None and len(exclude):
                row_doc = np.repeat(s.doc_ids, s.lengths)  # owning patent of each score
                scores = np.where(np.isin(row_doc, exclude), NEG_INF, scores)
            order = _top_order(scores, rows, k)
        return rows[order], scores[order], s.path

    def doc_row_scores(self, q, doc_id: int) -> tuple[np.ndarray, np.ndarray]:
        """(row ids, scores) for every chunk of one patent — cheap, used only for the final top-k."""
        a, b = int(self.doc_lo[doc_id]), int(self.doc_hi[doc_id])
        return np.arange(a, b), np.asarray(self.emb[a:b]) @ np.asarray(q, dtype=np.float32)
