"""JSON API over SearchEngine + a small test UI at `/` (also the unit the Part 2 PoC containerises).

Endpoints (interactive docs at /docs when the server runs):
  GET /                          the web UI (web/index.html)
  GET /search?q=...              text search; no q = list patents matching the filters
  GET /similar/{doc_number}      patent -> similar patents
  GET /claim/{doc}/{claim_no}    claim -> closest passages;  GET /claim?text=...  pasted claim text
  GET /patent/{doc_number}       one patent's metadata, claims and description
  GET /meta, /health             corpus facts for the UI; liveness check
All search endpoints accept the same filter parameters (cpc, kw, title, ... see filter_params).
"""

from __future__ import annotations

import asyncio
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from .embed import Embedder
from .filters import FilterSpec
from .search import SearchEngine, _fmt_date

Strategy = Literal["pre", "post", "post-topN"]
Granularity = Literal["patent", "passage"]
KwField = Literal["both", "title", "abstract"]

WEB_DIR = Path(__file__).resolve().parent / "web"
BENCH_QUERIES = Path(__file__).resolve().parent.parent / "bench" / "queries.jsonl"


def filter_params(
    cpc: list[str] | None = Query(None, description="CPC constraint(s), OR'ed: B60B, 'B60C 11', 'B60C 11/03'"),
    kw: list[str] | None = Query(None, description="keywords/phrases that must appear in title/abstract"),
    kw_any: bool = False,
    kw_field: KwField = "both",
    title: str | None = None,
    title_contains: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> FilterSpec:
    """FastAPI dependency: the filter query-string parameters -> FilterSpec.

    Declared once and reused by every search endpoint via `Depends(filter_params)`. Empty values
    (what an HTML form sends for blank fields) are treated as "not set".
    """
    clean = lambda xs: [x for x in (xs or []) if x and x.strip()]  # noqa: E731
    return FilterSpec(
        cpc=clean(cpc), keywords=clean(kw), kw_mode="any" if kw_any else "all", kw_field=kw_field,
        title=title or None, title_contains=title_contains or None,
        date_from=date_from or None, date_to=date_to or None,
    )


def create_app(
    index_dir: Path | None = None, embedder: Embedder | None = None, warm: bool = False, mmap: bool = True
) -> FastAPI:
    """Build the app around one SearchEngine. warm=True loads the model before serving."""
    app = FastAPI(title="patsearch", version="0.1.0")
    engine = SearchEngine(index_dir, embedder=embedder, mmap=mmap)
    # All engine work runs on ONE dedicated thread: it serialises access to the shared SQLite
    # connection and model, and MPS/torch pay a ~50-130 ms first-call cost per new thread, so a
    # single warmed-up thread keeps every query fast.
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="patsearch-engine")
    if warm:
        worker.submit(lambda: engine.search("warm up the model", k=1)).result()

    async def call(fn, *args, **kwargs):
        """Run blocking engine code on the worker thread (the event loop stays free meanwhile) and
        turn expected errors into HTTP errors: KeyError -> 404 not found, ValueError -> 400 bad input."""
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(worker, functools.partial(fn, *args, **kwargs))
        except KeyError as exc:
            raise HTTPException(404, exc.args[0] if exc.args else str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    async def run(fn, *args, **kwargs):
        """call() for engine methods that return a SearchResponse; returns its JSON-ready dict."""
        return await call(lambda: fn(*args, **kwargs).to_dict())

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    async def health():
        m = engine.store.manifest
        return {"status": "ok", "build_id": m["build_id"], "docs": m["counts"]["docs"], "chunks": m["counts"]["chunks"]}

    @app.get("/meta")
    async def meta():
        """Corpus facts for the UI: CPC suggestions, date range, example inputs."""
        con, m = engine.store.con, engine.store.manifest

        def q():
            groups = con.execute(
                "SELECT cpc_subclass, cpc_group, count(*) n FROM docs WHERE cpc_group IS NOT NULL "
                "GROUP BY cpc_group ORDER BY n DESC LIMIT 40"
            ).fetchall()
            return groups, con.execute("SELECT min(pub_date), max(pub_date) FROM docs").fetchone()

        groups, (lo, hi) = await call(q)
        examples: dict[str, list[str]] = {"text": [], "claim": [], "similar": []}
        if BENCH_QUERIES.exists():
            for line in BENCH_QUERIES.read_text().splitlines():
                q = json.loads(line)
                examples[q["type"]].append(q.get("text") or q.get("ref") or q.get("doc"))
        return {
            "build_id": m["build_id"],
            "model": m["model"]["id"],
            "docs": m["counts"]["docs"],
            "chunks": m["counts"]["chunks"],
            "subclasses": m["counts"]["cpc_subclasses"],
            "groups": [
                {"code": f"{g['cpc_subclass']} {g['cpc_group'][len(g['cpc_subclass']):]}", "n": g["n"]} for g in groups
            ],
            "date_min": _fmt_date(lo),
            "date_max": _fmt_date(hi),
            "examples": examples,
        }

    @app.get("/search")
    async def search(
        q: str | None = None,
        k: int = Query(10, ge=1, le=200),
        scope: str | None = None,
        granularity: Granularity = "patent",
        strategy: Strategy = "pre",
        spec: FilterSpec = Depends(filter_params),
    ):
        """Text search (q), or a filter-only listing when q is empty."""
        return await run(engine.search, q, spec, k=k, scope=scope or None, granularity=granularity, strategy=strategy)

    @app.get("/similar/{doc_number}")
    async def similar(
        doc_number: str,
        k: int = Query(10, ge=1, le=200),
        scope: str | None = None,
        spec: FilterSpec = Depends(filter_params),
    ):
        """Patents most similar to doc_number (aliases such as A9 numbers resolve to the A1)."""
        return await run(engine.similar, doc_number, spec, k=k, scope=scope or None)

    @app.get("/claim/{doc_number}/{claim_no}")
    async def claim(
        doc_number: str,
        claim_no: int,
        k: int = Query(10, ge=1, le=200),
        scope: str = "claims,description",
        granularity: Granularity = "patent",
        strategy: Strategy = "pre",
        spec: FilterSpec = Depends(filter_params),
    ):
        """Claim mapping from a stored claim: closest claims/paragraphs in other patents."""
        return await run(engine.claim, f"{doc_number}:{claim_no}", None, spec, k=k, scope=scope or None,
                   granularity=granularity, strategy=strategy)

    @app.get("/claim")
    async def claim_text(
        text: str = Query(..., min_length=3, description="pasted claim text"),
        k: int = Query(10, ge=1, le=200),
        scope: str = "claims,description",
        granularity: Granularity = "patent",
        strategy: Strategy = "pre",
        spec: FilterSpec = Depends(filter_params),
    ):
        """Claim mapping from pasted claim text (embedded with the model)."""
        return await run(engine.claim, None, text, spec, k=k, scope=scope or None, granularity=granularity, strategy=strategy)

    @app.get("/patent/{doc_number}")
    async def patent(doc_number: str):
        """Full record: metadata, abstract, re-assembled claims, description chunks."""
        s = engine.store

        def q():
            d = s.resolve(doc_number)
            if d is None:
                raise KeyError(f"unknown patent {doc_number!r}")
            desc = s.con.execute("SELECT label, text FROM chunks WHERE doc_id = ? AND kind = 2 ORDER BY chunk_id", (d,))
            return (s.docs([d])[d], [dict(c) for c in s.claims_of(d)], [dict(r) for r in desc],
                    s.aliases_of([d]).get(d, []))

        m, claims, desc, aliases = await call(q)
        return {
            "doc_number": m["doc_number"],
            "kind": m["kind"],
            "title": m["title"],
            "abstract": m["abstract"],
            "pub_date": _fmt_date(m["pub_date"]),
            "cpc": m["cpc_display"],
            "cpc_raw": m["cpc_raw"],
            "has_description": bool(m["has_description"]),
            "n_canceled": m["n_canceled"],
            "flags": json.loads(m["flags"]),
            "also_published_as": aliases,
            "claims": [
                {"lo": c["lo"], "hi": c["hi"], "label": c["label"], "exact": bool(c["exact"]),
                 "depends_on": c["depends_on"], "text": c["text"]}
                for c in claims
            ],
            "description": desc,
        }

    return app
