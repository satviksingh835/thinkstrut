# patsearch — hybrid semantic search over vehicle patent applications

Thinkstruct take-home. Part 1 (search engine with hybrid constraints + timings) is implemented here;
Parts 2 (system at scale) and 3 (enhancement: **Efficiency & large data**) build on it.

## Problem statement (Part 1)

Given ~640 US patent applications (vehicles, Feb 2024 – May 2025), let an examiner:

1. **search with a natural-language description** and get patents ranked by meaning, each with the
   specific **claim / description paragraph / abstract** that matched;
2. **start from a patent** (by publication number) and find the most similar other patents, with
   claim-to-passage evidence;
3. **start from one claim** ("claim mapping") and find the closest claims and paragraphs in other patents;

and combine any of these with **hard constraints** — CPC classification prefix (e.g. `B60B`), keywords
that must appear in the title/abstract, a specific title, a publication-date range — *without*
making search slower. Filtering is applied before vector scoring, so selective filters make search
faster (numbers below).

## Quick start

```bash
# 1. Python env (uv: https://docs.astral.sh/uv/) — pinned in uv.lock, Python 3.12
uv sync

# 2. Data: unzip the provided dataset so the JSON files are in data/patent_data_small/
#    (or pass --data-dir to `build`)
ls data/patent_data_small | head -2      # patents_ipa240215.json ...

# 3. Build the index (~2 min on an M4 using MPS; downloads BAAI/bge-small-en-v1.5 once, ~130 MB)
uv run patsearch build

# 4. Search
uv run patsearch search "tire with sound absorbing foam in the cavity to reduce noise"
uv run patsearch search "quick release for the wheel axle" --cpc B60B --kw bicycle
uv run patsearch search "noise" --title "pneumatic tire" --date-from 2025-01-01
uv run patsearch search --title TIRE -k 5          # filters only: newest first
uv run patsearch similar 20240131861               # patent -> similar patents
uv run patsearch claim 20240051333:1               # claim -> closest claims/paragraphs elsewhere
uv run patsearch claim --text "a spoke made of carbon fiber with threaded ends" --granularity passage
uv run patsearch show 20240051338                  # metadata + re-assembled claims
uv run patsearch search "hub" --json               # machine-readable response (incl. timings)

# 5. Tests (no model download needed) and benchmark
uv run pytest
uv run patsearch bench --synthetic 100000,1000000  # writes results/bench_part1.md

# 6. Web UI + JSON API (also what the Part 2 proof of concept containerises)
uv run patsearch serve    # open http://127.0.0.1:8000  ·  API docs at /docs
                          # GET /search?q=...&cpc=B60B&kw=bicycle, /similar/{doc}, /claim/{doc}/{n}, /claim?text=..., /patent/{doc}
```

**Test UI** (`patsearch/web/index.html`, plain HTML/JS served by the API): three modes (search,
similar patent, claim mapping), all hybrid filters (CPC/keyword chips, exact title, title phrase,
dates), scope/granularity/strategy options, evidence passages with keyword highlighting, a per-stage
timing bar for every query, a **Compare strategies** button (runs the same query with pre-filter /
naive post-filter / post-filter top-100 and tabulates median retrieval time and result overlap), and a
patent drawer with the re-assembled claims ("Map this claim" runs claim mapping from it). Clicking a
CPC code adds it as a filter. The form state lives in the URL hash, so reloads and shared links keep
the query; stale responses are discarded when queries overlap. All engine work runs on one warmed-up
worker thread (MPS pays a first-call cost per new thread).

Filter options (all commands): `--cpc` (repeatable, OR), `--kw` (repeatable, AND; `--kw-any` for OR;
`--kw-field title|abstract|both`; trailing `*` = prefix), `--title` (exact), `--title-contains`,
`--date-from/--date-to`. Search options: `-k`, `--scope abstract,claims,description`,
`--granularity patent|passage`, `--strategy pre|post|post-topN` (how filters are applied — see timings).

## Data: what was found and how it is handled

Profiling script: `scripts/explore_data.py <data_dir>`. The files don't match the spec exactly:
keys are lowercase (`title`, `doc_number`, `filename`, `abstract`, `detailed_description` (list),
`claims` (list), `bibtex`, `classification`), and there is an undocumented `filename` key that
carries the kind code and publication date.

| Finding | Handling |
|---|---|
| No key is ever absent or null, but **119 patents have `detailed_description == [""]`** (one empty string) | **Kept.** Flagged `has_description=0`, indexed on title/abstract/claims, shown as `[no description]`; `--scope description` never returns them. Excluding them would drop 18.6% of the corpus while their abstracts and claims are perfectly searchable. |
| ~49% of description paragraphs are `""` spacers | Dropped; paragraph labels (`¶12`) use the cleaned 1-based index. |
| `claims` holds **parser fragments**, not claims: nested claims lost their number and preamble (only sub-elements remain, unnumbered); canceled claims look like `"1 - 12 . (canceled)"`; claim numbers ≠ list positions in 1,131 cases | **Gap-aware re-assembly** (`patsearch/claims.py`): a numbered fragment starts that claim; unnumbered fragments fill the numbering gap before the next number. Result: 8,192 exactly-numbered claims, 340 ambiguous blocks (`claims 13–20`), 199 open tails (`claims 14+`), 1,031 canceled claims dropped. Asking for a claim number inside a block returns the block with a warning. |
| Claim 1 usually starts mid-sentence ("an axle body, having …"): preamble lost | The patent title is prepended to every chunk's *embedding* text (display text stays raw). |
| 4 pairs are byte-identical A1 + A9 (corrected) republications under different numbers | Indexed **once, under the A1** (earliest publication date, the one that matters for prior art); the A9 number is an alias — lookups by either number work, results show `[also 2024022743x]`. → **636 patents** (116 without description). |
| ~7 families share an abstract but have different claims | Kept as distinct patents; marked `[same family]` in `similar` results. |
| One classification string per patent, separators stripped + `FI` suffix: `B60C1103FI` = B60C 11/03 | Parsed into subclass / main group / subgroup (`patsearch/normalize.py`). A naive string prefix would be wrong (`B60C1` matches 11/03); see filter semantics below. 3 codes with ≥5 digits have an ambiguous split and are flagged. |
| Titles 89% ALL CAPS; Unicode math (′ ° ≤ θ), 403 invisible U+2062; `<`/`>` are math, not HTML | Case-insensitive matching; invisible chars stripped; `<`/`>` kept. |
| `bibtex` is fully derivable from the other fields | Stored, not indexed. Publication date comes from `filename`. |

The loader is tolerant beyond what this dataset needs (for Part 3's larger data): absent / null /
wrong-type fields become empty + a flag; a record is dropped only without a document number or with no
text at all (counts in `artifacts/<build>/manifest.json`; 0 here).

## Design

```text
data/*.json ──load──dedupe──parse(CPC, claims)──order──chunk──embed──┐
                                                                     ├─ artifacts/<build_id>/
     patents.sqlite: docs, aliases, claims, chunks(text), docs_fts (FTS5 title+abstract)
     chunks.f32.npy (24,057 × 384, memory-mapped) · docvec.f32.npy · row-range arrays · manifest.json

query ─┬─ text ──────── bge query embedding ─┐
       ├─ patent ID ─── stored vectors ──────┼─► FilterSpec ─► SQL + FTS5 ─► candidate doc ids
       └─ claim ref ─── stored claim vector ─┘                                   │
                                     ExactIndex: score only candidate rows (slices | gather | full)
                                     → best chunk per patent (np.maximum.reduceat) → top-k
                                     → evidence (best claim / paragraph / abstract) from SQLite
```

- **Chunks** (`chunk.py`): abstract, each re-assembled claim, and description paragraphs merged to
  ~250 tokens (split with overlap above 320). The build uses the model's tokenizer to check each
  input after adding the title, any parent-claim context, and special tokens. The default model
  truncates inputs over 512 tokens. The full passage remains visible as evidence, but text past
  the limit cannot contribute to its semantic search score. The manifest reports the number of
  over-limit inputs as `chunks_truncated` (0 in the supplied corpus build).
- **Model**: `BAAI/bge-small-en-v1.5` (384-d, 512 tokens, revision pinned). Small and fast, which
  fits the efficiency track; MiniLM truncates at 256 tokens, larger models were 3–10× slower.
- **Ranking**: patent score = its best chunk's cosine; evidence = the best chunk of each kind, so long
  descriptions don't crowd out claims. *Patent → similar*: stage 1 cosine between per-patent vectors
  (mean of abstract + claim chunks), stage 2 re-scores 50 candidates by max-sim (for each source
  claim/abstract chunk, its best match in the candidate, averaged) — evidence reads
  `claim 2 → ¶14`. *Claim → passages* reuses the stored claim vector (no model call).
- **Physical order**: `doc_id`s are assigned by (CPC subclass, publication date, number) and each
  patent's chunks are contiguous, so a subclass filter is one contiguous slice of the matrix, and a
  date range is one slice per subclass.
- **Tried and reverted**: prepending the parent claim to dependent claims' embedding text. It made each
  dependent claim a near-copy of its parent, so evidence pointed at dependent claims whose own text
  did not match the query (`ChunkConfig.parent_tokens`, default 0).

### Hybrid filter semantics

| Filter | Meaning | Implementation |
|---|---|---|
| `--cpc B60` / `B60B` | subclass prefix | B-tree range scan on `cpc_subclass` |
| `--cpc "B60C 11"` | main group 11 (not 1/xx) | equality on parsed `cpc_group` |
| `--cpc "B60C 11/03"` or `B60C1103` | exact code | equality on separator-free `cpc_compact` |
| `--kw tread --kw "sound absorber"` | every term appears in title or abstract (word/phrase, case- and diacritic-insensitive, no stemming; `groove*` = prefix) | FTS5 `{title abstract} : ("tread" AND "sound absorber")` |
| `--title "PNEUMATIC TIRE"` | exact title (case/punctuation-insensitive) | equality on indexed `title_norm` |
| `--title-contains hub` | phrase in title | FTS5 column filter |
| `--date-from / --date-to` | publication date range | B-tree on `pub_date` |

User text never reaches FTS5 verbatim: terms are re-tokenised to alphanumerics and quoted, so input
like `B60-B`, `"`, `NEAR(` can't cause an FTS5 syntax error (tested). The naive post-filter baseline uses
a Python predicate designed to match SQLite's filtering semantics. See [Known bugs](#known-bugs)
for a Unicode case where the two paths differ.

## Timing: with vs without hybrid search

Full tables: [`results/bench_part1.md`](results/bench_part1.md) (`uv run patsearch bench --synthetic 100000,1000000`).
Apple M4, numpy/Accelerate, 200 warm reps per cell over 20 fixed queries (`bench/queries.jsonl`),
k = 10. Retrieval = filter + score + aggregate + top-k; query embedding is reported separately.
(The timings printed by a single CLI call include one-off warm-up — first MPS inference, paging in
the memory-mapped matrix — so they are higher than these steady-state numbers.)

**Real index (636 patents, 24,057 vectors):**

| filter | rows kept | **pre-filter** (path) | naive post-filter (exhaustive) | ANN-style post-filter on top-100 (recall@10) |
|---|---|---|---|---|
| none | 100% | **0.53 ms** (full) | — | — |
| `cpc=B60C` | 48.5% | **0.29 ms** (slices) | 0.60 ms | 0.60 ms (0.97) |
| `cpc=B60D` | 3.3% | **0.033 ms** (slices) | 0.76 ms | 0.57 ms (**0.14**) |
| `pub_date ≥ 2025-02-01` | 23.6% | **0.093 ms** (slices) | 0.56 ms | 0.56 ms (1.0) |
| `kw=bicycle` | 6.0% | **0.074 ms** (slices) | 21.9 ms | 8.1 ms (**0.26**) |
| `title=TIRE` | 5.9% | **0.083 ms** (slices) | 0.57 ms | 0.55 ms (0.61) |
| `cpc=B60B & kw=bicycle` | 5.8% | **0.081 ms** (slices) | 6.4 ms | 2.5 ms (**0.24**) |

Query embedding (bge-small on MPS): **6.0 ms p50**, so a full text query end-to-end is ≈ 7 ms; the
`similar` and claim-by-reference inputs reuse stored vectors and take < 1 ms end-to-end.

**Synthetic scale-up** (random unit vectors with the real chunk-count and CPC distributions):

| rows | none | `cpc=B60C` (51%) | `cpc=B60D` (2.9%) | scattered 5% | scattered 0.5% | naive post-filter | post-top100 recall (2.9% / 5% / 0.5%) |
|---|---|---|---|---|---|---|---|
| 100k | 2.18 ms | 1.08 (slices) | 0.03 (slices) | 0.23 (gather) | 0.04 | ~2.3 ms | 0.33 / 0.56 / 0.03 |
| 1M | 21.9 ms | 11.2 (slices) | 0.66 (slices) | 4.5 (gather) | 0.23 (gather) | ~23–26 ms | 0.38 / 0.57 / 0.06 |

### Commentary: making hybrid search fast

- **Filter first, with indexes, then score only the survivors.** The constraints are cheap to evaluate
  against indexes (B-trees for CPC/title/date, the FTS5 inverted index for keywords: 0.01–0.08 ms), and
  the expensive part (vector scoring) is proportional to the surviving rows. So hybrid search is
  *faster* than plain search: 0.03–0.29 ms vs 0.53 ms here, and 0.23–11 ms vs 22 ms at 1M rows.
- **Lay data out so common filters are contiguous.** Ordering patents by (CPC subclass, date) turns
  subclass and date filters into a few contiguous ranges that are scored as zero-copy slices. Forcing a
  gather (copying the same rows) costs 1.7–4.3× more across scales, and at 1M rows a 51% gather is
  even slower than scoring everything (37.6 vs 21.9 ms) — hence the cost-based path choice.
- **Naive post-filtering is slow and/or wrong.** Exhaustive post-filtering pays full scoring plus a
  per-candidate check; a per-candidate keyword check is the worst case (21.9 ms at 636 patents vs
  0.07 ms with the inverted index). Post-filtering an ANN's top-N is fast but loses results exactly when
  filters are selective (recall 0.14 for `B60D`, 0.06 at 0.5% selectivity).
- **Skip the model when possible.** Patent- and claim-based queries reuse stored vectors (saves the
  6 ms embedding, which is otherwise the dominant cost at this size).
- **What I'd add at scale** (Part 2/3): an ANN index (HNSW/IVF) with *filter-aware* search — exact
  brute force on the candidate subset below ~1–5% selectivity (0.66 ms for 29k rows), ANN with in-index
  filtering (FAISS `IDSelector`, Qdrant/pgvector filtered search) above it — rather than
  post-filtering; partitioning the vector index by CPC subclass so a CPC filter prunes whole
  partitions; roaring bitmaps for candidate sets; query-embedding caching. The 1M-row scan reads
  1.5 GB per query in 22 ms (~70 GB/s): it is memory-bandwidth bound — with one BLAS thread
  (`VECLIB_MAXIMUM_THREADS=1`, [`results/bench_part1_1thread.md`](results/bench_part1_1thread.md)) it takes
  22.1 ms, i.e. more threads don't help — so fp16/int8 vectors (2–4× less to read) are the next lever,
  together with batching queries into matrix-matrix products.

## Part 2 — system design at 10 M patents

[`docs/part2_system_design.md`](docs/part2_system_design.md): components (S3, Postgres, OpenSearch,
batch workers, API), the ingestion and query pipelines, a cost breakdown (~$50–200 one-off backfill,
~$2.5–4 k/month), error handling, status tracking, and the main challenges. Key decision, backed by a
measurement on this corpus: index every chunk (378 M vectors, 1-bit compressed) rather than one
summary vector per patent, which missed ~20% of the true top-10 when re-scoring 20 candidates and
4–8% even with 100 (16% of the corpus).

**Proof of concept** ([`poc/`](poc/README.md)): the ingestion pipeline and job queue of the design,
running locally with Docker: PostgreSQL job queue (`FOR UPDATE SKIP LOCKED`), a scheduler, three
worker containers running the six stages with the Part 1 code, and a status dashboard. With injected
failures, it verified all 64 weekly files, recovered 71 transient errors and 10 worker crashes,
parked a corrupt file in `needs_review`, and loaded the same 636 patents as Part 1.

```bash
docker compose -f poc/docker-compose.yml up --build -d   # then open http://localhost:8080
```

## Part 3 — Efficiency and large data

Chosen enhancement: efficiency and large data, because the Part 1 pipeline would need about 86 hours
to embed the 471,000-patent sample on this laptop and its exact search slows linearly with the corpus.
Full report: [`results/part3_efficiency.md`](results/part3_efficiency.md).

- **Ingestion 1.93× faster, 2.6× less memory**: a streaming build (`patsearch build --profile fast`)
  with a CPU process pool overlapping float16 GPU embedding, deduplication before embedding and
  resumable shards (lossless: agreement@10 with the baseline 0.995–1.0), plus an optional, measured
  lossy step that embeds only the first 64 description chunks (agreement@10 0.88–0.94).
- **Unfiltered search 17.6× faster** at 640k vectors (14.6 ms to 0.83 ms, recall@10 0.985) with a FAISS
  HNSW index (`patsearch ann-index`) that applies filters inside the graph search; selective filters keep
  the exact path.
- Problems found at scale and fixed: duplicate records, dateless filenames, a claim-numbering bug, a
  torch/faiss OpenMP conflict and memory pressure from unbounded prefetching.

```bash
uv run patsearch build --profile fast --max-desc-chunks 64 --data-dir data/coding_challenge_data --limit-files 10
uv run patsearch ann-index
uv run patsearch search "lithium battery cathode material"
```

## Limitations

- Retrieval quality is not formally evaluated (the chosen Part 3 track is efficiency); `bge-small` is a
  general model and the claim text is fragmented upstream. Best-chunk ranking slightly favours long
  patents (more chunks, more chances).
- The CPC main-group/subgroup split for 5+ digit codes is heuristic (3 codes; flagged in the index).
- Ambiguous claim blocks keep their text searchable, but the original claim boundaries are unknown.
- The build holds the (small) corpus in memory; Part 3 turns ingestion into a streaming,
  batch-at-a-time pipeline with an embedding cache.

## Repository layout

```text
patsearch/   load.py normalize.py claims.py chunk.py tokens.py embed.py build.py store.py
             filters.py vindex.py search.py bench.py cli.py api.py timing.py config.py
tests/       unit + engine tests on a synthetic corpus (HashEmbedder, no download) + real-data profile check
scripts/     explore_data.py (data profiling)
bench/       queries.jsonl (fixed benchmark queries)
results/     bench_part1.md (benchmark report)
docs/        part2_system_design.md; architecture diagram source (diagrams/architecture.py) and image (images/)
poc/         Part 2 proof of concept: docker-compose, schema.sql, pipeline/ (job queue, stages, worker, scheduler, dashboard)
```

## Known bugs

- **Keyword filters can disagree on Unicode text.** A keyword filter for `strasse` matches a title
  containing `Straße` in the Python post-filter. Python's `casefold()` converts `ß` to `ss`.
  The SQLite FTS5 pre-filter uses the `unicode61` tokenizer, which preserves that distinction.
  Switching strategies can therefore change which patents pass the keyword constraint. To fix this,
  align tokenization across both paths and add a regression test for this example.
