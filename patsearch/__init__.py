"""patsearch — hybrid semantic search over patent applications.

Map of the code (roughly in the order data flows through it):

  Indexing — `patsearch build`, runs once (~2 min)
    load.py       read the weekly JSON files -> Patent objects (tolerant of missing fields); dedupe
    normalize.py  text cleanup, title normalisation, CPC classification-code parsing
    claims.py     re-assemble the fragmented `claims` lists into numbered claims
    chunk.py      split each patent into chunks: abstract, each claim, description paragraphs
    tokens.py     count model tokens so every chunk fits the model's 512-token window
    embed.py      chunk text -> 384-number vectors (the embedding model)
    store.py      on-disk layout: SQLite (text, metadata, keyword index) + numpy matrices (vectors)
    build.py      runs the steps above in order and times each one

  Searching — `patsearch search | similar | claim`, the JSON API and the web UI
    filters.py    hard constraints (CPC, keywords, title, dates) -> the set of allowed patents
    vindex.py     cosine similarity between a query vector and the allowed chunk vectors
    search.py     the engine: query -> vector -> filter -> score -> rank patents -> evidence
    cli.py, api.py, web/index.html   the three front ends over search.py

  Measuring
    timing.py     per-stage stopwatch;  bench.py  with-vs-without-filter benchmark
"""

__version__ = "0.1.0"

import os as _os

# faiss (ANN index, Part 3) and torch (embedding model) each bundle their own OpenMP runtime on macOS.
# Whichever library initialises OpenMP second aborts the process ("OMP: Error #15"), regardless of
# import order. The established remedy is to allow the duplicate runtime; it must be set before either
# library starts. Correctness is checked by tests/test_ann.py (ANN results identical to exact search)
# and by the Part 3 benchmark (recall@10 against exact search).
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
