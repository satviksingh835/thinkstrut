"""Search engine: text query, patent -> similar patents, claim -> closest passages.

What one text search does, start to finish (see SearchEngine.search):
  1. embed the query            text -> one 384-number vector              (embed.py, ~6 ms)
  2. filter                     SQL + FTS5 -> doc_ids allowed by the hard constraints (filters.py)
  3. score                      cosine of the query vs every allowed chunk vector     (vindex.py)
  4. aggregate + rank           patent score = its best chunk's score; keep the top k
  5. hydrate                    fetch titles, metadata and evidence text for those k (SQLite)

Filter strategies (hybrid constraints):
  pre        filter first (SQL/FTS5 -> doc ids), score only those docs' rows   [default]
  post       naive: score everything, walk the full ranking, check each doc with a Python predicate
  post-topN  what "ANN + filter afterwards" does: filter only the unfiltered top-N (may under-fill)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .chunk import KIND_ABSTRACT, KIND_CLAIM, KIND_CODES, KIND_DESCRIPTION, KIND_NAMES, KIND_TITLE
from .embed import Embedder, load_embedder
from .filters import CompiledFilter, DocPredicate, FilterSpec, candidate_doc_ids, compile_filter, load_doc_meta
from .store import Store, resolve_artifact_dir
from .timing import Stopwatch
from .vindex import ExactIndex, _top_order

ALL_KINDS = (KIND_ABSTRACT, KIND_CLAIM, KIND_DESCRIPTION, KIND_TITLE)
STRATEGIES = ("pre", "post", "post-topN")
POST_TOPN = 100  # candidate-list size for the "post-topN" baseline (what an ANN index would return)


@dataclass
class Passage:
    """One evidence passage (a chunk) shown under a hit, with its cosine score."""

    chunk_id: int
    kind: str
    label: str
    score: float
    text: str
    matched_from: str | None = None  # similar(): which part of the source patent matched this passage


@dataclass
class Hit:
    """One ranked patent. score is None in browse mode (filters only, no query to score against)."""

    rank: int
    doc_number: str
    title: str
    score: float | None
    pub_date: str | None
    cpc: str | None
    has_description: bool
    also_published_as: list[str] = field(default_factory=list)
    same_family: bool = False
    evidence: list[Passage] = field(default_factory=list)


@dataclass
class SearchResponse:
    """Everything a query returns; to_dict() is the JSON the API sends and `--json` prints.

    selectivity_docs / selectivity_rows: share of patents / chunk vectors that passed the filters.
    path: how the vector scoring was done (full / slices / gather, see vindex.py).
    """

    mode: str
    query: dict
    filters: dict
    k: int
    strategy: str
    path: str | None
    n_docs: int
    n_candidates: int | None  # docs passing the filters (None when not computed, e.g. post strategies)
    selectivity_docs: float | None
    selectivity_rows: float | None
    hits: list[Hit]
    timings_ms: dict
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_scope(scope: str | list[str] | tuple | None) -> set[int] | None:
    """'claims,description' -> {1, 2}. None/'all' -> None (every kind)."""
    if scope is None:
        return None
    items = scope.split(",") if isinstance(scope, str) else list(scope)
    kinds = set()
    for it in items:
        it = it.strip().lower().rstrip("s")  # claims -> claim
        if it in ("", "all"):
            return None
        if it not in KIND_CODES:
            raise ValueError(f"unknown scope {it!r}; use abstract, claims, description")
        kinds.add(KIND_CODES[it])
    if KIND_ABSTRACT in kinds:
        kinds.add(KIND_TITLE)  # title-only fallback chunks travel with the abstract
    return kinds


def _fmt_date(d: int | None) -> str | None:
    """20240215 -> '2024-02-15'."""
    if not d:
        return None
    s = str(d)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def parse_claim_ref(ref: str) -> tuple[str, int]:
    """'20240051333:1', '20240051333#1', '20240051333 claim 1' -> ('20240051333', 1)."""
    m = re.match(r"^\s*([A-Za-z0-9/ ]+?)\s*(?:[:#]|\s+claim\s*)\s*(\d+)\s*$", ref, re.I)
    if not m:
        raise ValueError(f"bad claim reference {ref!r}; expected DOC_NUMBER:CLAIM_NO, e.g. 20240051333:1")
    return m.group(1), int(m.group(2))


class SearchEngine:
    """Opens a built index and answers queries. One instance serves many queries.

    The embedding model is loaded lazily (on the first text query): patent- and claim-based
    queries reuse vectors stored at build time and never need it.
    """

    def __init__(self, artifact_dir: Path | None = None, embedder: Embedder | None = None, mmap: bool = True,
                 ann: bool = True):
        self.store = Store(resolve_artifact_dir(artifact_dir), mmap=mmap)
        s = self.store
        self.exact = ExactIndex(s.emb, s.doc_chunk_lo, s.doc_chunk_hi, s.chunk_kind)
        self.index = self.exact
        if ann:  # use the HNSW index when the build has one (Part 3); falls back to exact search
            try:
                from .ann import AnnIndex
                self.index = AnnIndex.load(self.exact, self.store.dir) or self.exact
            except ImportError:
                pass
        self._embedder = embedder
        self._meta = None
        self._abstract_hash: list[str | None] | None = None

    # --- lazy resources -------------------------------------------------------------------------
    @property
    def embedder(self) -> Embedder:
        """The model, loaded on first use. Must be the model the index was built with: vectors
        from different models live in different spaces and can't be compared."""
        if self._embedder is None:
            self._embedder = load_embedder(self.store.manifest["model"]["id"])
        if self._embedder.model_id != self.store.manifest["model"]["id"]:
            raise RuntimeError(
                f"embedder {self._embedder.model_id} != index model {self.store.manifest['model']['id']}"
            )
        return self._embedder

    @property
    def doc_meta(self):
        """Per-patent metadata in memory — only needed by the naive post-filter strategies."""
        if self._meta is None:
            self._meta = load_doc_meta(self.store.con)
        return self._meta

    @property
    def n_docs(self) -> int:
        return self.index.n_docs

    def embed_query(self, text: str) -> np.ndarray:
        return self.embedder.encode_queries([text])[0]

    # --- filtering ------------------------------------------------------------------------------
    def select(self, cf: CompiledFilter) -> np.ndarray | None:
        """Pre-filter: doc_ids passing the constraints, or None when there are no constraints."""
        return None if cf.is_empty else candidate_doc_ids(self.store.con, cf)

    def _selectivity(self, doc_ids: np.ndarray | None) -> tuple[int, float, float]:
        """(patents kept, share of patents kept, share of chunk vectors kept) — shown in responses."""
        if doc_ids is None:
            return self.n_docs, 1.0, 1.0
        n = int(doc_ids.size)
        return n, n / max(self.n_docs, 1), self.index.selected_rows(doc_ids) / max(self.index.n_rows, 1)

    # --- core ranking with strategy selection ----------------------------------------------------
    def rank_docs(
        self,
        q: np.ndarray,
        cf: CompiledFilter,
        k: int,
        kinds: set[int] | None,
        exclude: list[int],
        strategy: str,
        path: str,
        sw: Stopwatch,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Top-k patents for query vector q under the filters, using the chosen strategy.

        All three strategies return the same patents when they can (tests check pre == post);
        they differ in how much work they do, which is what the benchmark measures.
        """
        info: dict = {"path": None, "n_candidates": None, "sel_docs": None, "sel_rows": None}
        excl = np.asarray(exclude, dtype=np.int64)
        # --- pre-filter: ask the indexes which patents qualify, then score only those --------
        if strategy == "pre" or cf.is_empty:
            with sw.stage("filter"):
                doc_ids = self.select(cf)
            info["n_candidates"], info["sel_docs"], info["sel_rows"] = self._selectivity(doc_ids)
            ids, scores, used = self.index.top_docs(q, k, doc_ids, kinds, excl, path, sw)
            info["path"] = used
            return ids, scores, info

        pred = DocPredicate(cf, self.doc_meta)
        # --- naive post-filter: score and rank everything, then check patents one by one -------
        if strategy == "post":
            ids, best, used = self.index.doc_scores(q, None, kinds, "full", sw)
            with sw.stage("topk"):
                if excl.size:
                    best = best.copy()
                    best[np.isin(ids, excl)] = -np.inf
                order = np.lexsort((ids, -best))
            with sw.stage("post_filter"):
                keep = []
                for i in order:
                    if best[i] == -np.inf:
                        break
                    if pred(int(ids[i])):
                        keep.append(i)
                        if len(keep) == k:
                            break
            info["path"] = "full"
            keep_arr = np.asarray(keep, dtype=np.int64)
            return ids[keep_arr], best[keep_arr], info

        # --- ANN-style post-filter: filter only a fixed-size unfiltered shortlist --------------
        # Fast, but if fewer than k of the top-N pass the filter, results are silently missing.
        if strategy == "post-topN":
            top_ids, top_scores, used = self.index.top_docs(q, POST_TOPN, None, kinds, excl, "full", sw)
            with sw.stage("post_filter"):
                mask = np.fromiter((pred(int(d)) for d in top_ids), dtype=bool, count=top_ids.size)
            info["path"] = f"full+top{POST_TOPN}"
            return top_ids[mask][:k], top_scores[mask][:k], info

        raise ValueError(f"unknown strategy {strategy!r}; use one of {STRATEGIES}")

    # --- hydration ------------------------------------------------------------------------------
    def _evidence(self, q: np.ndarray, doc_ids, kinds: set[int] | None, n: int, mode: str) -> dict[int, list]:
        """Pick evidence rows per doc. mode='per_kind': best row of each kind; 'top': best n rows.

        Runs only for the final top-k patents, so re-scoring their chunks here costs almost nothing.
        'per_kind' (text search) shows the best claim, paragraph and abstract separately, so a long
        description can't crowd the claims out of the evidence.
        """
        picked: dict[int, list[tuple[int, float]]] = {}
        for d in doc_ids:
            rows, scores = self.index.doc_row_scores(q, int(d))
            krow = self.index.chunk_kind[rows]
            ok = np.ones(rows.size, bool) if kinds is None else np.isin(krow, list(kinds))
            rows, scores, krow = rows[ok], scores[ok], krow[ok]
            if mode == "per_kind":
                best = []
                for kd in np.unique(krow):
                    m = krow == kd
                    j = int(np.argmax(scores[m]))
                    best.append((int(rows[m][j]), float(scores[m][j])))
                best.sort(key=lambda t: -t[1])
            else:
                order = np.argsort(-scores, kind="stable")[:n]
                best = [(int(rows[j]), float(scores[j])) for j in order]
            picked[int(d)] = best[:n]
        return picked

    def _hits(
        self,
        doc_ids,
        scores,
        evidence: dict[int, list] | None,
        matched_from: dict[int, str] | None = None,
        family_hash: str | None = None,
    ) -> list[Hit]:
        """Build Hit objects: fetch metadata, aliases and evidence text from SQLite in a few batched
        queries (only for the k results — candidates never touch the database's text)."""
        ids = [int(d) for d in doc_ids]
        meta = self.store.docs(ids, "doc_id, doc_number, title, pub_date, cpc_display, has_description, abstract_hash")
        aliases = self.store.aliases_of(ids)
        chunk_ids = [c for d in ids for c, _ in (evidence or {}).get(d, [])]
        chunks = self.store.chunks(chunk_ids)
        hits = []
        for rank, (d, sc) in enumerate(zip(ids, scores), start=1):
            m = meta[d]
            ev = []
            for cid, s in (evidence or {}).get(d, []):
                c = chunks[cid]
                ev.append(
                    Passage(cid, KIND_NAMES[c["kind"]], c["label"], round(s, 4), c["text"], (matched_from or {}).get(cid))
                )
            hits.append(
                Hit(
                    rank=rank,
                    doc_number=m["doc_number"],
                    title=m["title"],
                    score=None if sc is None else round(float(sc), 4),
                    pub_date=_fmt_date(m["pub_date"]),
                    cpc=m["cpc_display"],
                    has_description=bool(m["has_description"]),
                    also_published_as=aliases.get(d, []),
                    same_family=bool(family_hash and m["abstract_hash"] == family_hash),
                    evidence=ev,
                )
            )
        return hits

    def _response(self, mode, query, cf, k, strategy, info, hits, sw, warnings=()) -> SearchResponse:
        return SearchResponse(
            mode=mode,
            query=query,
            filters=cf.spec.to_dict(),
            k=k,
            strategy=strategy,
            path=info.get("path"),
            n_docs=self.n_docs,
            n_candidates=info.get("n_candidates"),
            selectivity_docs=None if info.get("sel_docs") is None else round(info["sel_docs"], 4),
            selectivity_rows=None if info.get("sel_rows") is None else round(info["sel_rows"], 4),
            hits=hits,
            timings_ms=sw.as_dict(),
            warnings=list(cf.warnings) + list(warnings),
        )

    # ==============================================================================================
    # Public API
    # ==============================================================================================
    def search(
        self,
        query: str | None,
        filters: FilterSpec | None = None,
        k: int = 10,
        scope: str | list[str] | None = None,
        granularity: str = "patent",
        strategy: str = "pre",
        path: str = "auto",
        n_evidence: int = 3,
        query_vec: np.ndarray | None = None,
        hydrate: bool = True,
    ) -> SearchResponse:
        """Natural-language search (or filter-only browse when `query` is empty)."""
        sw = Stopwatch()
        cf = compile_filter(filters)
        kinds = parse_scope(scope)
        if not (query and query.strip()) and query_vec is None:
            return self.browse(cf, k, sw)
        if query_vec is None:
            self.embedder  # load the model outside the timed stage (one-off cost, not per query)
            with sw.stage("embed_query"):
                query_vec = self.embed_query(query)
        return self._run(
            "text", {"text": query}, query_vec, cf, k, kinds, [], granularity, strategy, path, n_evidence,
            "per_kind", sw, hydrate,
        )

    def _run(self, mode, query, q, cf, k, kinds, exclude, granularity, strategy, path, n_evidence, ev_mode, sw,
            hydrate, warnings=()):
        """Shared second half of search() and claim(): rank with query vector q, then hydrate.

        granularity='patent' ranks patents (best chunk each); 'passage' ranks individual chunks.
        """
        warnings = list(warnings)
        if granularity == "passage":
            if strategy != "pre":
                warnings.append("passage granularity always uses the pre-filter strategy")
                strategy = "pre"
            with sw.stage("filter"):
                doc_ids = self.select(cf)
            n_c, sd, sr = self._selectivity(doc_ids)
            rows, scores, used = self.index.top_rows(q, k, doc_ids, kinds, np.asarray(exclude, np.int64), path, sw)
            info = {"path": used, "n_candidates": n_c, "sel_docs": sd, "sel_rows": sr}
            hits: list[Hit] = []
            if hydrate:
                with sw.stage("hydrate"):
                    triples = list(zip(rows.tolist(), scores.tolist(), self.store.chunk_doc[rows].tolist()))
                    docs = list(dict.fromkeys(d for _, _, d in triples))
                    ev: dict[int, list] = {}
                    for r, s, d in triples:
                        ev.setdefault(d, []).append((r, s))
                    by_doc = {h_doc: h for h_doc, h in zip(docs, self._hits(docs, [None] * len(docs), ev))}
                    # one Hit per passage, in passage order
                    for i, (r, s, d) in enumerate(triples, start=1):
                        base = by_doc[d]
                        ev_one = [p for p in base.evidence if p.chunk_id == r]
                        hits.append(replace(base, rank=i, score=round(s, 4), evidence=ev_one))
            return self._response(mode, query, cf, k, strategy, info, hits, sw, warnings)

        ids, scores, info = self.rank_docs(q, cf, k, kinds, exclude, strategy, path, sw)
        hits = []
        if hydrate:
            with sw.stage("hydrate"):
                ev = self._evidence(q, ids, kinds, n_evidence, ev_mode)
                hits = self._hits(ids, scores, ev)
        return self._response(mode, query, cf, k, strategy, info, hits, sw, warnings)

    def browse(self, cf: CompiledFilter, k: int, sw: Stopwatch | None = None) -> SearchResponse:
        """Filters only, no query: newest first."""
        sw = sw or Stopwatch()
        if cf.is_empty:
            raise ValueError("give a query, or at least one filter")
        with sw.stage("filter"):
            doc_ids = self.select(cf)
        n_c, sd, sr = self._selectivity(doc_ids)
        with sw.stage("hydrate"):
            meta = self.store.docs(doc_ids.tolist(), "doc_id, doc_number, pub_date")
            order = sorted(meta.values(), key=lambda r: (-(r["pub_date"] or 0), r["doc_number"]))[:k]
            hits = self._hits([r["doc_id"] for r in order], [None] * len(order), None)
        info = {"path": None, "n_candidates": n_c, "sel_docs": sd, "sel_rows": sr}
        return self._response("browse", {}, cf, k, "pre", info, hits, sw)

    def similar(
        self,
        doc_number: str,
        filters: FilterSpec | None = None,
        k: int = 10,
        scope: str | list[str] | None = None,
        n_candidates: int = 50,
        n_evidence: int = 3,
        hydrate: bool = True,
    ) -> SearchResponse:
        """Patent -> similar patents. No model call: uses stored vectors.

        Stage 1: cosine between per-patent vectors (mean of abstract + claim chunks), filters applied.
        Stage 2: re-score the top `n_candidates` by max-sim — for every abstract/claim chunk of the
        source take its best-matching chunk in the candidate, average. The best pairs are evidence
        ("source claim 1 ~ candidate ¶12").
        """
        sw = Stopwatch()
        cf = compile_filter(filters)
        kinds = parse_scope(scope)
        d = self.store.resolve(doc_number)
        if d is None:
            raise KeyError(f"unknown patent {doc_number!r}")
        idx = self.index
        # The source patent's own abstract + claim vectors (stored at build time) act as the query:
        # claims define the invention, so they are what "similar" should be judged on.
        src_rows = np.arange(idx.doc_lo[d], idx.doc_hi[d])
        src_kinds = idx.chunk_kind[src_rows]
        sel = np.isin(src_kinds, [KIND_ABSTRACT, KIND_CLAIM])
        if sel.any():
            src_rows = src_rows[sel]
        Q = np.asarray(self.store.emb[src_rows])  # (source chunks x 384)

        with sw.stage("filter"):
            doc_ids = self.select(cf)
        n_c, sd, sr = self._selectivity(doc_ids)
        # Stage 1 (cheap, all allowed patents): one summary vector per patent vs the source's.
        with sw.stage("stage1_docvec"):
            ids = idx.all_doc_ids if doc_ids is None else doc_ids
            s1 = self.store.docvec[ids] @ self.store.docvec[d]
            s1 = np.where(ids == d, -np.inf, s1).astype(np.float32)  # never return the patent itself
            cand = ids[_top_order(s1, ids, n_candidates)]
        # Stage 2 (precise, 50 candidates only): compare chunk by chunk. For each source chunk find
        # its best match in the candidate (row-wise max of the similarity matrix M), then average —
        # "how well is every part of the source covered by this candidate?"
        with sw.stage("stage2_maxsim"):
            scored = []
            for c in cand.tolist():
                rows = np.arange(idx.doc_lo[c], idx.doc_hi[c])
                if kinds is not None:
                    rows = rows[np.isin(idx.chunk_kind[rows], list(kinds))]
                    if rows.size == 0:
                        continue
                M = Q @ np.asarray(self.store.emb[rows]).T  # (source chunks, candidate chunks)
                best_j = M.argmax(axis=1)
                best = M[np.arange(M.shape[0]), best_j]
                # evidence: strongest (source chunk -> candidate chunk) pairs, one per candidate chunk
                pairs, used = [], set()
                for i in np.argsort(-best, kind="stable"):
                    cid = int(rows[best_j[i]])
                    if cid not in used:
                        used.add(cid)
                        pairs.append((cid, float(best[i]), int(src_rows[i])))
                        if len(pairs) == n_evidence:
                            break
                scored.append((float(best.mean()), c, pairs))
            scored.sort(key=lambda t: (-t[0], t[1]))
            scored = scored[:k]
        hits: list[Hit] = []
        if hydrate:
            with sw.stage("hydrate"):
                src_meta = self.store.docs([d], "doc_id, doc_number, title, abstract_hash")[d]
                ev = {c: [(cid, s) for cid, s, _ in pairs] for _, c, pairs in scored}
                src_chunks = self.store.chunks([sid for _, _, pairs in scored for _, _, sid in pairs])
                matched_from = {cid: src_chunks[sid]["label"] for _, _, pairs in scored for cid, _, sid in pairs}
                hits = self._hits(
                    [c for _, c, _ in scored], [s for s, _, _ in scored], ev, matched_from, src_meta["abstract_hash"]
                )
        query = {"doc_number": self.store.docs([d], "doc_id, doc_number")[d]["doc_number"], "source_vectors": int(Q.shape[0])}
        info = {"path": "docvec+maxsim", "n_candidates": n_c, "sel_docs": sd, "sel_rows": sr}
        return self._response("similar", query, cf, k, "pre", info, hits, sw)

    def claim(
        self,
        ref: str | None = None,
        text: str | None = None,
        filters: FilterSpec | None = None,
        k: int = 10,
        scope: str | list[str] | None = "claims,description",
        granularity: str = "patent",
        strategy: str = "pre",
        path: str = "auto",
        n_evidence: int = 3,
        exclude_source: bool = True,
        hydrate: bool = True,
    ) -> SearchResponse:
        """Claim -> closest claims / description passages in other patents ("claim mapping")."""
        sw = Stopwatch()
        cf = compile_filter(filters)
        kinds = parse_scope(scope)
        warnings: list[str] = []
        exclude: list[int] = []
        if text:
            self.embedder  # load the model outside the timed stage
            with sw.stage("embed_query"):
                q = self.embedder.encode_passages([text])[0]  # passage<->passage: no query prefix
            query = {"text": text}
        elif ref:
            doc_number, n = parse_claim_ref(ref)
            d = self.store.resolve(doc_number)
            if d is None:
                raise KeyError(f"unknown patent {doc_number!r}")
            unit = next(
                (u for u in self.store.claims_of(d) if u["lo"] <= n and (u["hi"] is None or n <= u["hi"])), None
            )
            if unit is None:
                raise KeyError(f"patent {doc_number} has no claim {n} (it may be canceled)")
            if not unit["exact"]:
                warnings.append(
                    f"claim {n} lies in '{unit['label']}' (numbering lost upstream); the whole block is used"
                )
            # reuse the claim's stored vector (averaged if a long claim was split into parts, then
            # re-normalised to length 1) — no model call needed
            cids = self.store.claim_chunk_ids(d, unit["lo"])
            v = np.asarray(self.store.emb[cids]).mean(axis=0)
            q = (v / max(float(np.linalg.norm(v)), 1e-12)).astype(np.float32)
            query = {"doc_number": doc_number, "claim": n, "label": unit["label"], "text": unit["text"]}
            if exclude_source:
                exclude = [d]  # the claim trivially matches its own patent; show other patents
        else:
            raise ValueError("give a claim reference (DOC:CLAIM) or --text")
        return self._run(
            "claim", query, q, cf, k, kinds, exclude, granularity, strategy, path, n_evidence, "top", sw, hydrate,
            warnings,
        )
