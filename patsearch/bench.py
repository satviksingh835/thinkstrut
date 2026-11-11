"""Benchmark: search latency with vs without hybrid filters, per strategy, plus synthetic scale-up.

Methodology
  * Query vectors are precomputed; model inference (query embedding) is timed separately because
    it dominates at this corpus size and does not depend on the filters.
  * Each (scenario, strategy) cell runs `reps` calls cycling over the query set after `warmup`
    calls; we report p50/p95 of retrieval time (filter + score + aggregate + top-k + post-filter)
    and the p50 of the main stages. Hydration (SQLite fetch of the top-k rows) is reported once.
  * `post-topN` is what "ANN top-N, then filter" does; its recall@k vs the exact pre-filter is shown.
  * The synthetic section scales the matrix to N rows (random unit vectors, per-doc chunk counts
    and CPC subclasses drawn from the real index, docs contiguous by subclass) so that the effect of
    filter selectivity on scoring cost is visible — at 24k rows everything is ~1 ms.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from .filters import DocPredicate, FilterSpec, compile_filter
from .search import POST_TOPN, SearchEngine
from .timing import Stopwatch
from .vindex import NEG_INF, ExactIndex

QUERIES_FILE = Path(__file__).resolve().parent.parent / "bench" / "queries.jsonl"
K = 10  # results per query

# Filter settings with very different selectivity (share of patents they keep): from none, through
# ~50% (a big subclass), down to ~3% and keyword/title filters that keep ~6%.
SCENARIOS: list[tuple[str, FilterSpec]] = [
    ("none", FilterSpec()),
    ("cpc=B60C", FilterSpec(cpc=["B60C"])),
    ("cpc=B60B", FilterSpec(cpc=["B60B"])),
    ("cpc=B60D", FilterSpec(cpc=["B60D"])),
    ("pub_date>=2025-02-01", FilterSpec(date_from="2025-02-01")),
    ("kw=bicycle", FilterSpec(keywords=["bicycle"])),
    ("title=TIRE", FilterSpec(title="TIRE")),
    ("cpc=B60B & kw=bicycle", FilterSpec(cpc=["B60B"], keywords=["bicycle"])),
]
# Each scenario is run with each of these. The "force" rows pin the scoring path to show what the
# automatic choice (vindex.ExactIndex.choose_path) saves.
STRATEGY_CELLS = [  # (label, strategy, forced path)
    ("pre (auto path)", "pre", "auto"),
    ("pre, force gather", "pre", "gather"),
    ("pre, force full+mask", "pre", "full"),
    ("post (exhaustive)", "post", "auto"),
    (f"post-top{POST_TOPN}", "post-topN", "auto"),
]


def _pct(xs, p):
    """p-th percentile (50 = median, 95 = p95 tail latency)."""
    return float(np.percentile(np.asarray(xs), p)) if len(xs) else float("nan")


def _machine() -> str:
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = ""
    return f"{cpu or platform.processor()} · {platform.system()} {platform.machine()} · numpy {np.__version__}"


def load_queries(path: Path = QUERIES_FILE) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------------
# real index
# ------------------------------------------------------------------------------------------------
def _query_vectors(eng: SearchEngine, queries: list[dict], reps: int, log) -> tuple[dict, list[np.ndarray]]:
    """Returns embedding-latency stats and vectors for single-vector searches (text + claim refs)."""
    texts = [q["text"] for q in queries if q["type"] == "text"]
    eng.embedder  # load model (not timed)
    for t in texts[:3]:
        eng.embed_query(t)  # warm-up (MPS kernels)
    lat = []
    for i in range(max(reps // 4, 20)):
        t0 = time.perf_counter_ns()
        eng.embed_query(texts[i % len(texts)])
        lat.append((time.perf_counter_ns() - t0) / 1e6)
    vecs = [eng.embed_query(t) for t in texts]
    for q in queries:
        if q["type"] == "claim":  # stored claim vector, exactly as SearchEngine.claim() builds it
            doc, n = q["ref"].split(":")
            d = eng.store.resolve(doc)
            unit = next(u for u in eng.store.claims_of(d) if u["lo"] <= int(n) and (u["hi"] is None or int(n) <= u["hi"]))
            cids = eng.store.claim_chunk_ids(d, unit["lo"])
            v = np.asarray(eng.store.emb[cids]).mean(0)
            vecs.append((v / np.linalg.norm(v)).astype(np.float32))
    stats = {
        "device": getattr(eng.embedder, "device", None),
        "p50_ms": round(_pct(lat, 50), 2),
        "p95_ms": round(_pct(lat, 95), 2),
        "n": len(lat),
    }
    log(f"query embedding: p50 {stats['p50_ms']} ms, p95 {stats['p95_ms']} ms on {stats['device']}")
    return stats, vecs


def bench_real(eng: SearchEngine, vecs: list[np.ndarray], reps: int, warmup: int, log) -> list[dict]:
    """Time rank_docs() for every scenario x strategy on the real index, and check result quality:
    recall@k = share of the exact pre-filter top-k that the strategy also returned."""
    rows = []
    for name, spec in SCENARIOS:
        cf = compile_filter(spec)
        ids = eng.select(cf)
        n_c, sd, sr = eng._selectivity(ids)
        # reference answers: the exact pre-filter top-k for every query
        pre_ref = [eng.rank_docs(v, cf, K, None, [], "pre", "auto", Stopwatch())[0].tolist() for v in vecs]
        cells = STRATEGY_CELLS if not cf.is_empty else STRATEGY_CELLS[:1]
        for label, strategy, path in cells:
            for i in range(warmup):
                eng.rank_docs(vecs[i % len(vecs)], cf, K, None, [], strategy, path, Stopwatch())
            totals, stage = [], {}
            used_path = None
            for i in range(reps):
                sw = Stopwatch()
                _, _, info = eng.rank_docs(vecs[i % len(vecs)], cf, K, None, [], strategy, path, sw)
                used_path = info["path"]
                totals.append(sw.total_ms)
                for k_, v in sw.ms.items():
                    stage.setdefault(k_, []).append(v)
            recalls, identical = [], True
            for v, ref in zip(vecs, pre_ref):
                got = eng.rank_docs(v, cf, K, None, [], strategy, path, Stopwatch())[0].tolist()
                identical &= got == ref
                recalls.append(len(set(got) & set(ref)) / max(len(ref), 1))
            row = {
                "scenario": name,
                "docs_kept": n_c,
                "sel_docs": round(sd, 4),
                "sel_rows": round(sr, 4),
                "strategy": label,
                "path": used_path,
                "p50_ms": round(_pct(totals, 50), 3),
                "p95_ms": round(_pct(totals, 95), 3),
                "stage_p50_ms": {k_: round(_pct(v, 50), 3) for k_, v in stage.items()},
                "recall_at_k": round(float(np.mean(recalls)), 3),
                "identical_to_pre": identical,
            }
            rows.append(row)
            log(f"  {name:<24} {label:<22} {row['path'] or '':<14} p50 {row['p50_ms']:.3f} ms  recall {row['recall_at_k']}")
    return rows


def bench_similar(eng: SearchEngine, docs: list[str], reps: int, warmup: int, log) -> list[dict]:
    """Time patent -> similar (stage 1 + stage 2) with and without filters."""
    out = []
    for name, spec in [s for s in SCENARIOS if s[0] in ("none", "cpc=B60B", "cpc=B60D", "kw=bicycle")]:
        for i in range(warmup):
            eng.similar(docs[i % len(docs)], spec, k=K, hydrate=False)
        totals, s1, s2 = [], [], []
        for i in range(reps):
            r = eng.similar(docs[i % len(docs)], spec, k=K, hydrate=False)
            totals.append(r.timings_ms["total"])
            s1.append(r.timings_ms.get("stage1_docvec", 0))
            s2.append(r.timings_ms.get("stage2_maxsim", 0))
        row = {"scenario": name, "p50_ms": round(_pct(totals, 50), 3), "p95_ms": round(_pct(totals, 95), 3),
               "stage1_p50_ms": round(_pct(s1, 50), 3), "stage2_p50_ms": round(_pct(s2, 50), 3)}
        out.append(row)
        log(f"  similar {name:<14} p50 {row['p50_ms']:.3f} ms")
    return out


def bench_end_to_end(eng: SearchEngine, queries: list[dict], reps: int, log) -> list[dict]:
    """What a user of the CLI/API sees per call (model already loaded), incl. embedding + hydration."""
    by_type = {
        "text": [lambda q=q: eng.search(q["text"], FilterSpec(cpc=["B60C"]), k=K) for q in queries if q["type"] == "text"],
        "claim (by ref)": [lambda q=q: eng.claim(q["ref"], k=K) for q in queries if q["type"] == "claim"],
        "similar": [lambda q=q: eng.similar(q["doc"], k=K) for q in queries if q["type"] == "similar"],
    }
    out = []
    for name, fns in by_type.items():
        for f in fns[:3]:
            f()
        totals, hyd = [], []
        n = max(reps // 4, 20)
        for i in range(n):
            r = fns[i % len(fns)]()
            totals.append(r.timings_ms["total"])
            hyd.append(r.timings_ms.get("hydrate", 0))
        out.append({"input": name, "p50_ms": round(_pct(totals, 50), 2), "p95_ms": round(_pct(totals, 95), 2),
                    "hydrate_p50_ms": round(_pct(hyd, 50), 2)})
        log(f"  end-to-end {name:<16} p50 {out[-1]['p50_ms']} ms")
    return out


# ------------------------------------------------------------------------------------------------
# synthetic scale-up
# ------------------------------------------------------------------------------------------------
def _synthetic(eng: SearchEngine, n_rows: int, dim: int, seed: int = 0):
    """A fake index with `n_rows` chunk vectors that is shaped like the real one.

    Vectors are random (we measure cost, not relevance); chunks-per-patent and the CPC mix are
    sampled from the real index, and patents are sorted by subclass just like a real build.
    """
    rng = np.random.default_rng(seed)
    real_len = eng.index.lengths
    sub = [r[0] or "~" for r in eng.store.con.execute("SELECT cpc_subclass FROM docs ORDER BY doc_id")]
    # draw chunk counts per fake patent until they add up to n_rows (trim the last one to fit)
    lengths = rng.choice(real_len, size=int(n_rows / real_len.mean() * 1.05) + 10)
    cum = np.cumsum(lengths)
    n_docs = int(np.searchsorted(cum, n_rows)) + 1
    lengths = lengths[:n_docs]
    lengths[-1] -= int(lengths.sum() - n_rows)
    if lengths[-1] <= 0:
        lengths = lengths[:-1]
        lengths[-1] += n_rows - int(lengths.sum())
    n_docs = lengths.size
    subclasses = np.sort(rng.choice(np.asarray(sub), size=n_docs))  # contiguous by subclass
    hi = np.cumsum(lengths)
    lo = hi - lengths
    emb = np.empty((n_rows, dim), dtype=np.float32)
    step = 200_000  # generate in blocks to avoid a second full-size temporary array
    for a in range(0, n_rows, step):
        block = rng.standard_normal((min(step, n_rows - a), dim), dtype=np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        emb[a : a + block.shape[0]] = block
    kinds = rng.choice(eng.store.chunk_kind, size=n_rows).astype(np.int8)
    return ExactIndex(emb, lo, hi, kinds), subclasses


def bench_synthetic(eng: SearchEngine, sizes: list[int], reps: int, warmup: int, log) -> list[dict]:
    """Same strategy comparison on synthetic indexes of the given sizes (e.g. 100k and 1M rows).

    Filters are simulated as sets of doc_ids: a subclass (contiguous block) or a random scattered
    sample (like a keyword filter). The naive post-filter's check here is a cheap array lookup,
    which flatters it — the real keyword check is far slower (see the real-index table).
    """
    out = []
    dim = int(eng.store.emb.shape[1])
    rng = np.random.default_rng(1)
    for n in sizes:
        t0 = time.time()
        idx, subclasses = _synthetic(eng, n, dim)
        log(f"synthetic {n:,} rows / {idx.n_docs:,} docs generated in {time.time() - t0:.1f}s")
        qs = rng.standard_normal((10, dim), dtype=np.float32)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        all_ids = idx.all_doc_ids
        scen = [
            ("none", None),
            ("cpc=B60C", all_ids[subclasses == "B60C"]),
            ("cpc=B60D", all_ids[subclasses == "B60D"]),
            ("scattered 5% (keyword-like)", np.sort(rng.choice(all_ids, size=max(1, idx.n_docs // 20), replace=False))),
            ("scattered 0.5%", np.sort(rng.choice(all_ids, size=max(1, idx.n_docs // 200), replace=False))),
        ]
        for name, ids in scen:
            mask = np.zeros(idx.n_docs, bool)  # mask[d] = does patent d pass the simulated filter
            mask[all_ids if ids is None else ids] = True

            def pre(q, path="auto"):  # filter first: score only the allowed patents
                return idx.top_docs(q, K, ids, None, None, path)

            def post(q):  # score everything, sort all patents, take the first k that pass
                d, best, _ = idx.doc_scores(q, None, None, "full")
                order = np.lexsort((d, -best))
                keep = []
                for i in order:
                    if best[i] == NEG_INF:
                        break
                    if mask[d[i]]:
                        keep.append(int(d[i]))
                        if len(keep) == K:
                            break
                return keep

            def post_topn(q):  # unfiltered top-100, then drop those that fail (may return < k)
                d, _, _ = idx.top_docs(q, POST_TOPN, None, None, None, "full")
                return [int(x) for x in d if mask[x]][:K]

            fns = {"pre (auto path)": pre} if ids is None else {
                "pre (auto path)": pre,
                "pre, force gather": lambda q: pre(q, "gather"),
                "post (exhaustive)": post,
                f"post-top{POST_TOPN}": post_topn,
            }
            ref = [pre(q)[0].tolist() for q in qs]
            for label, f in fns.items():
                for i in range(warmup):
                    f(qs[i % len(qs)])
                lat = []
                for i in range(reps):
                    t = time.perf_counter_ns()
                    f(qs[i % len(qs)])
                    lat.append((time.perf_counter_ns() - t) / 1e6)
                got = [list(r[0]) if isinstance(r, tuple) else r for r in (f(q) for q in qs)]
                rec = float(np.mean([len(set(g) & set(r)) / max(len(r), 1) for g, r in zip(got, ref)]))
                used = idx.choose_path(ids, "auto") if label == "pre (auto path)" else ("gather" if "gather" in label else "full")
                row = {
                    "rows": n, "docs": idx.n_docs, "scenario": name,
                    "sel_rows": round(idx.selected_rows(ids) / n, 4), "strategy": label, "path": used,
                    "p50_ms": round(_pct(lat, 50), 2), "p95_ms": round(_pct(lat, 95), 2), "recall_at_k": round(rec, 3),
                }
                out.append(row)
                log(f"  {n:>9,} {name:<28} {label:<20} {used:<7} p50 {row['p50_ms']:8.2f} ms  recall {row['recall_at_k']}")
        del idx
    return out


# ------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------
def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    """Markdown table; each column is (header, dict key or function of the row)."""
    head = "| " + " | ".join(h for h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for r in rows:
        vals = []
        for _, key in cols:
            v = key(r) if callable(key) else r.get(key)
            vals.append("" if v is None else (f"{v:.1%}" if isinstance(v, float) and key in ("sel_docs", "sel_rows") else str(v)))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(res: dict, out_md: Path) -> None:
    """Render the benchmark results as the markdown report (results/bench_part1.md)."""
    m = res["meta"]
    parts = [
        "# Part 1 benchmark — search latency with vs without hybrid filters",
        "",
        f"- machine: {m['machine']}",
        f"- index: build `{m['build_id']}`, {m['n_docs']} patents, {m['n_rows']:,} chunk vectors × {m['dim']} dims "
        f"(float32, {m['matrix_mb']:.0f} MB), model `{m['model']}`",
        f"- BLAS threads: VECLIB_MAXIMUM_THREADS={m['veclib_threads']}; reps per cell {m['reps']} (after {m['warmup']} warm-up calls), "
        f"k={K}; times are wall-clock `perf_counter_ns`",
        "",
        "## Query embedding (model inference, independent of filters)",
        "",
        f"p50 **{res['embedding']['p50_ms']} ms**, p95 {res['embedding']['p95_ms']} ms on `{res['embedding']['device']}` "
        "(text queries; patent→similar and claim-by-reference reuse stored vectors and need no model call).",
        "",
        "## Retrieval on the real index (filter + score + aggregate + top-k), text & claim query vectors",
        "",
        _md_table(res["real"], [
            ("scenario", "scenario"), ("docs kept", "docs_kept"), ("rows kept", "sel_rows"), ("strategy", "strategy"),
            ("path", "path"), ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"),
            ("filter p50", lambda r: r["stage_p50_ms"].get("filter")),
            ("score p50", lambda r: r["stage_p50_ms"].get("score")),
            ("post-filter p50", lambda r: r["stage_p50_ms"].get("post_filter")),
            ("recall@10 vs pre", "recall_at_k"), ("same top-10 as pre", "identical_to_pre"),
        ]),
        "",
        "## Patent → similar (stage 1 doc vectors, stage 2 max-sim over 50 candidates), no hydration",
        "",
        _md_table(res["similar"], [("scenario", "scenario"), ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"),
                                   ("stage 1 p50", "stage1_p50_ms"), ("stage 2 p50", "stage2_p50_ms")]),
        "",
        "## End-to-end per call (model loaded; includes query embedding and SQLite hydration of results + evidence)",
        "",
        _md_table(res["end_to_end"], [("input", "input"), ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"),
                                      ("hydrate p50", "hydrate_p50_ms")]),
    ]
    if res.get("synthetic"):
        parts += [
            "",
            "## Synthetic scale-up (random unit vectors; real per-patent chunk counts and CPC mix; docs contiguous by subclass)",
            "",
            _md_table(res["synthetic"], [
                ("rows", lambda r: f"{r['rows']:,}"), ("scenario", "scenario"), ("rows kept", "sel_rows"),
                ("strategy", "strategy"), ("path", "path"), ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"),
                ("recall@10 vs pre", "recall_at_k"),
            ]),
        ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts) + "\n")


def run_benchmark(index_dir=None, reps=200, warmup=5, synthetic_sizes=(), out_md=Path("results/bench_part1.md"), log=print):
    """Entry point of `patsearch bench`: run every section, write the .md report and a .json dump."""
    eng = SearchEngine(index_dir, mmap=False)  # matrix in RAM: measure compute, not page faults
    queries = load_queries()
    log(f"[bold]benchmark[/bold] on {eng.store.dir.name}: {eng.n_docs} docs, {eng.index.n_rows:,} rows")
    emb_stats, vecs = _query_vectors(eng, queries, reps, log)
    log("retrieval on the real index:")
    real = bench_real(eng, vecs, reps, warmup, log)
    log("patent -> similar:")
    sim = bench_similar(eng, [q["doc"] for q in queries if q["type"] == "similar"], reps, warmup, log)
    log("end-to-end:")
    e2e = bench_end_to_end(eng, queries, reps, log)
    synth = bench_synthetic(eng, list(synthetic_sizes), max(20, reps // 5), 3, log) if synthetic_sizes else []
    man = eng.store.manifest
    res = {
        "meta": {
            "machine": _machine(),
            "build_id": man["build_id"],
            "model": man["model"]["id"],
            "n_docs": eng.n_docs,
            "n_rows": eng.index.n_rows,
            "dim": int(eng.store.emb.shape[1]),
            "matrix_mb": eng.store.emb.nbytes / 1e6,
            "veclib_threads": os.environ.get("VECLIB_MAXIMUM_THREADS", "default"),
            "reps": reps,
            "warmup": warmup,
        },
        "embedding": emb_stats,
        "real": real,
        "similar": sim,
        "end_to_end": e2e,
        "synthetic": synth,
    }
    write_report(res, Path(out_md))
    Path(out_md).with_suffix(".json").write_text(json.dumps(res, indent=2))
    log(f"wrote {out_md} and {Path(out_md).with_suffix('.json')}")
    return res
