# Technical Details

This document describes how each part of the project is implemented and measured. For the problem
statement and the instructions to reproduce the results, see the [README](README.md).

All measurements were taken on an Apple M4 (10 cores, 16 GB memory, MPS GPU) with macOS and
Python 3.12.

## Contents

- [Part 1: Search Engine](#part-1-search-engine)
  - [1.1 Data Analysis and Handling](#11-data-analysis-and-handling)
  - [1.2 Index Build](#12-index-build)
  - [1.3 Storage Layout](#13-storage-layout)
  - [1.4 Query Types and Ranking](#14-query-types-and-ranking)
  - [1.5 Hybrid Filters](#15-hybrid-filters)
  - [1.6 Timing With and Without Hybrid Search](#16-timing-with-and-without-hybrid-search)
  - [1.7 Making Hybrid Search Efficient](#17-making-hybrid-search-efficient)
  - [1.8 Interfaces](#18-interfaces)
  - [1.9 Code Map](#19-code-map)
  - [1.10 Limitations and Known Issues](#110-limitations-and-known-issues)
- [Part 2: Implementation at Scale](#part-2-implementation-at-scale)
  - [2.1 System Design](#21-system-design)
  - [2.2 Proof of Concept](#22-proof-of-concept)
- [Part 3: Efficiency and Large Data](#part-3-efficiency-and-large-data)
  - [3.1 Large Data Sample](#31-large-data-sample)
  - [3.2 Method](#32-method)
  - [3.3 Ingestion: Lossless Improvements](#33-ingestion-lossless-improvements)
  - [3.4 Ingestion: Description Limit](#34-ingestion-description-limit)
  - [3.5 Search: HNSW Index](#35-search-hnsw-index)
  - [3.6 Problems Found at Scale](#36-problems-found-at-scale)
  - [3.7 Extrapolation to the Full Sample](#37-extrapolation-to-the-full-sample)
  - [3.8 Result Files](#38-result-files)

## Part 1: Search Engine

### 1.1 Data Analysis and Handling

The sample contains 640 records in 64 weekly files. The profiling script is
`scripts/explore_data.py <data_dir>`.

The files differ from the task description in two ways:

- **Keys.** They are lowercase: `title`, `doc_number`, `abstract`, `detailed_description` (a list),
  `claims` (a list), `bibtex` and `classification`.
- **Extra field.** An undocumented `filename` field carries the kind code and the publication date.

| Finding | Handling |
|---|---|
| No field is absent or null, but 119 patents have `detailed_description == [""]` | Kept with `has_description = 0`, indexed on title, abstract and claims, and shown as having no description. A description-only search never returns them. Excluding them would remove 18.6% of the corpus, although their abstracts and claims are searchable. |
| About 49% of description paragraphs are empty spacers | Removed. Paragraph labels (`¶12`) use the 1-based index after cleaning. |
| `claims` holds parser fragments rather than claims. Nested claims lost their number and preamble. Canceled claims appear as `"1 - 12 . (canceled)"`. In 1,131 cases the claim number differs from the list position. | Gap-aware re-assembly (`patsearch/claims.py`). A numbered fragment starts a claim, and unnumbered fragments fill the gap before the next number. The result is 8,192 exactly numbered claims, 340 ambiguous blocks (`claims 13–20`) and 199 open tails (`claims 14+`); 1,031 canceled claims are removed. A request for a claim inside a block returns the block with a warning. |
| Claim 1 usually begins mid-sentence because its preamble was lost | The patent title is prepended to the embedding text of every chunk. The displayed text is unchanged. |
| 4 pairs of byte-identical publications under different numbers (A1 and a corrected A9) | Indexed once under the A1, which has the earliest publication date and is the relevant one for prior art. The A9 number is an alias: lookups by either number work, and results show `[also 2024022743x]`. 636 patents remain, 116 of them without a description. |
| About 7 patent families share an abstract but have different claims | Kept as separate patents and marked `[same family]` in similar-patent results |
| One classification string per patent, without separators and with an `FI` suffix: `B60C1103FI` is B60C 11/03 | Parsed into subclass, main group and subgroup (`patsearch/normalize.py`). A plain string prefix would be wrong: `B60C1` also matches 11/03. For 3 codes with five or more digits the split is ambiguous; they are flagged. |
| 89% of titles are in capitals; text contains Unicode symbols (′ ° ≤ θ), 403 invisible U+2062 characters, and `<`/`>` used as mathematical symbols | Matching is case-insensitive, invisible characters are removed, and `<`/`>` are kept |
| `bibtex` can be derived from the other fields | Stored but not indexed. The publication date is taken from `filename`. |

The loader also handles defects that this sample does not contain but the large data does:

- An absent, null or wrongly typed field becomes empty, and the record is flagged.
- A record is dropped only if it has no document number or no text.
- Counts of flagged and dropped records are saved in `artifacts/<build_id>/manifest.json`. No records
  were dropped from the sample.

### 1.2 Index Build

`patsearch build --profile baseline` (`patsearch/build.py`) runs the following stages and times each
one: load, deduplicate, parse classification codes and claims, order, chunk, embed, write.

- **Chunks** (`patsearch/chunk.py`). Each patent produces:
  - the abstract;
  - each re-assembled claim;
  - its description paragraphs, merged to about 250 tokens and split with overlap above 320 tokens.
- **Token limit.** Each input is checked with the model's tokenizer after the title and special
  tokens are added. Inputs over the model's 512-token limit would be truncated. The manifest reports
  their number as `chunks_truncated` (0 for the sample).
- **Model.** `BAAI/bge-small-en-v1.5` (384 dimensions, 512-token limit, pinned revision) on the MPS
  GPU. It is small and fast. MiniLM truncates at 256 tokens, and larger models were 3–10 times slower.
- **Result.** 636 patents and 24,057 chunks (641 abstract, 9,184 claim and 14,232 description chunks).
  The build takes 129 s, of which 118 s is embedding at 31,900 tokens/s.
- **Tried and reverted.** Prepending the parent claim to the embedding text of dependent claims. Each
  dependent claim became a near copy of its parent, so evidence pointed to dependent claims whose own
  text did not match the query. The option remains as `ChunkConfig.parent_tokens` (default 0).

### 1.3 Storage Layout

Each build writes `artifacts/<build_id>/`, and `artifacts/LATEST` names the most recent build.

| File | Content |
|---|---|
| `patents.sqlite` | Tables `docs`, `aliases`, `claims` and `chunks` (text); FTS5 table `docs_fts` over title and abstract; B-tree indexes on CPC fields, normalised title and publication date |
| `chunks.f32.npy` | Chunk vectors (24,057 × 384, float32), memory-mapped at search time |
| `docvec.f32.npy` | One vector per patent: the mean of its abstract and claim chunks |
| Row-range arrays | First and last chunk row of each patent, and the kind of each chunk |
| `manifest.json` | Build parameters, counts, data-quality statistics and stage timings |

Patent IDs are assigned in order of CPC subclass, publication date and number, and the chunks of
each patent are stored contiguously. A subclass filter therefore selects one contiguous range of the
vector matrix, and a date range selects one range per subclass.

### 1.4 Query Types and Ranking

| Input | Query vector | Ranking | Evidence |
|---|---|---|---|
| Text (`search`) | Query embedding with the model (6.0 ms) | Patent score = cosine similarity of its best chunk (`np.maximum.reduceat` over each patent's rows) | Best chunk of each kind (claim, paragraph, abstract), so long descriptions do not displace claims |
| Patent number (`similar`) | Stored patent vector | Stage 1 ranks patents by patent vector. Stage 2 re-scores 50 candidates by max-sim: for each source claim or abstract chunk, its best match in the candidate, averaged. | Pairs of source and matched passage, such as `claim 2 → ¶14` |
| Claim (`claim`) | Stored claim vector (no model call) or embedded claim text | As for text; the claim's own patent is excluded | Closest claims and paragraphs; `--granularity passage` returns passages instead of patents |

A query that has filters but no text returns the matching patents, newest first.

### 1.5 Hybrid Filters

| Filter | Meaning | Implementation |
|---|---|---|
| `--cpc B60` or `B60B` | Subclass prefix | B-tree range scan on `cpc_subclass` |
| `--cpc "B60C 11"` | Main group 11 (not 1/xx) | Equality on the parsed `cpc_group` |
| `--cpc "B60C 11/03"` or `B60C1103` | Exact code | Equality on the separator-free `cpc_compact` |
| `--kw tread --kw "sound absorber"` | Every term appears in the title or abstract. Matching is by word or phrase, ignores case and diacritics, and does not stem; `groove*` matches a prefix. | FTS5 query `{title abstract} : ("tread" AND "sound absorber")` |
| `--title "PNEUMATIC TIRE"` | Exact title, ignoring case and punctuation | Equality on the indexed `title_norm` |
| `--title-contains hub` | Phrase in the title | FTS5 column filter |
| `--date-from`, `--date-to` | Publication date range | B-tree on `pub_date` |

User text never reaches FTS5 unchanged. Terms are re-tokenised to alphanumeric characters and
quoted, so input such as `B60-B`, `"` or `NEAR(` cannot cause an FTS5 syntax error; this is tested.

Filters are compiled into a set of patent IDs (`patsearch/filters.py`). `ExactIndex`
(`patsearch/vindex.py`) then chooses one of three scoring paths by cost:

| Path | Used when | Method |
|---|---|---|
| `slices` | The selection consists of a few contiguous ranges | Each range is scored as a zero-copy slice of the matrix |
| `gather` | The selection is scattered and small | The selected rows are copied and scored |
| `full` | The selection is large | All rows are scored and the rest are masked |

The `--strategy` option selects how filters are applied, for comparison:

- `pre`: filter first (the default);
- `post`: score everything, then filter;
- `post-topN`: filter only the top N results, as a post-filtered approximate index would.

### 1.6 Timing With and Without Hybrid Search

**Method.** The full results are in [results/bench_part1.md](results/bench_part1.md)
(`uv run patsearch bench --synthetic 100000,1000000`).

- Each cell is the median of 200 warm runs over 20 fixed queries (`bench/queries.jsonl`), with k = 10.
- Retrieval time covers filtering, scoring, aggregation and top-k selection. Query embedding is
  reported separately.
- A single CLI call also includes one-time warm-up costs, so it reports higher times.

**Sample index** (636 patents, 24,057 vectors). The last column filters the top 100 results, as a
post-filtered approximate index would; its recall@10 is measured against the filter-first result.

| Filter | Rows kept | Filter first (path) | Naive post-filter | Post-filter of top 100 (recall@10) |
|---|---|---|---|---|
| None | 100% | 0.53 ms (full) | | |
| `cpc=B60C` | 48.5% | 0.29 ms (slices) | 0.60 ms | 0.60 ms (0.97) |
| `cpc=B60D` | 3.3% | 0.033 ms (slices) | 0.76 ms | 0.57 ms (0.14) |
| `pub_date ≥ 2025-02-01` | 23.6% | 0.093 ms (slices) | 0.56 ms | 0.56 ms (1.0) |
| `kw=bicycle` | 6.0% | 0.074 ms (slices) | 21.9 ms | 8.1 ms (0.26) |
| `title=TIRE` | 5.9% | 0.083 ms (slices) | 0.57 ms | 0.55 ms (0.61) |
| `cpc=B60B` and `kw=bicycle` | 5.8% | 0.081 ms (slices) | 6.4 ms | 2.5 ms (0.24) |

Query embedding takes 6.0 ms (median). A text query therefore takes about 7 ms end to end, including
result assembly. Similar-patent and claim queries reuse stored vectors and take less than 1 ms end to
end.

**Synthetic indexes.** Random unit vectors with the real distributions of chunks per patent and of
CPC subclasses.

| Rows | None | `cpc=B60C` (51%) | `cpc=B60D` (2.9%) | Scattered 5% | Scattered 0.5% | Naive post-filter | Post-filter of top 100, recall@10 (2.9% / 5% / 0.5%) |
|---|---|---|---|---|---|---|---|
| 100,000 | 2.18 ms | 1.08 ms | 0.03 ms | 0.23 ms | 0.04 ms | about 2.3 ms | 0.33 / 0.56 / 0.03 |
| 1,000,000 | 21.9 ms | 11.2 ms | 0.66 ms | 4.5 ms | 0.23 ms | 23–26 ms | 0.38 / 0.57 / 0.06 |

### 1.7 Making Hybrid Search Efficient

- **Filter first through indexes, then score only the selection.** The constraints are evaluated
  through B-trees and the FTS5 inverted index in 0.01–0.08 ms. The cost of vector scoring is
  proportional to the number of selected rows. Hybrid search is therefore faster than unfiltered
  search: 0.03–0.29 ms against 0.53 ms on the sample, and 0.23–11 ms against 22 ms at one million rows.
- **Store data so that common filters select contiguous ranges.** Ordering patents by CPC subclass
  and date turns subclass and date filters into a few slices that are scored without copying. Copying
  the same rows instead costs 1.7–4.3 times more. At one million rows, copying a 51% selection
  (37.6 ms) is slower than scoring every row (21.9 ms), which is why the path is chosen by cost.
- **Avoid naive post-filtering.** An exhaustive post-filter pays for full scoring plus a check of
  every candidate. A per-candidate keyword check is the worst case: 21.9 ms on the sample, against
  0.07 ms with the inverted index. Post-filtering the top N of an approximate index is fast but loses
  results when filters are selective (recall@10 of 0.14 for `B60D` and 0.06 at 0.5% selectivity).
- **Avoid the model where possible.** Patent and claim queries reuse stored vectors, which saves the
  6 ms embedding step, the largest cost at this corpus size.
- **Further measures** (applied in Part 3 or designed in Part 2):
  - an approximate index with filters applied inside the index search;
  - partitioning the vector index by CPC subclass;
  - compressed bitmaps for candidate sets;
  - a cache for query embeddings.

  The one-million-row scan reads 1.5 GB in 22 ms (about 70 GB/s), so it is limited by memory
  bandwidth. With one BLAS thread it takes 22.1 ms
  ([results/bench_part1_1thread.md](results/bench_part1_1thread.md)), so more threads do not help.
  Float16 or 8-bit vectors, which reduce the data read by two to four times, are the next measure.

### 1.8 Interfaces

| Interface | Entry point | Content |
|---|---|---|
| CLI | `patsearch/cli.py` | Commands `build`, `search`, `similar`, `claim`, `show`, `bench`, `ann-index` and `serve` |
| HTTP API | `patsearch/api.py` (FastAPI) | `GET /search`, `/similar/{doc}`, `/claim/{doc}/{n}`, `/claim?text=…` and `/patent/{doc}`; documentation at `/docs` |
| Web interface | `patsearch/web/index.html` | Plain HTML and JavaScript, served by the API |

The web interface offers:

- the three query types, all filters, and the scope, granularity and strategy options;
- evidence passages with highlighted keywords, and a timing breakdown for each query;
- a strategy comparison that runs one query with each filter strategy and tabulates median time and
  result overlap;
- a patent view with the re-assembled claims, from which any claim can be mapped.

The form state is kept in the URL, and responses to superseded requests are discarded.

### 1.9 Code Map

| Path | Module | Responsibility |
|---|---|---|
| Indexing | `load.py` | Read JSON files into `Patent` objects, tolerating missing fields; deduplicate |
| | `normalize.py` | Text cleaning, title normalisation, CPC parsing |
| | `claims.py` | Claim re-assembly |
| | `chunk.py`, `tokens.py` | Chunking within the model's token limit |
| | `embed.py` | Embedding model wrapper (float32 or float16) |
| | `build.py` | Part 1 build (`--profile baseline`) |
| | `fastbuild.py` | Part 3 streaming build (default profile) |
| | `store.py` | On-disk layout: SQLite and NumPy arrays |
| Search | `filters.py` | Filters to the set of allowed patents |
| | `vindex.py` | Exact vector scoring with cost-based paths |
| | `ann.py` | Part 3 HNSW index with in-graph filtering |
| | `search.py` | Query engine: query, vector, filter, score, rank, evidence |
| | `cli.py`, `api.py`, `web/` | User interfaces |
| Measurement | `timing.py`, `bench.py` | Per-stage timing; benchmark |

The tests in `tests/` (115) use a synthetic corpus and a model-free `HashEmbedder`, so they need no
model download. They include a profile check of the real data, which runs when the data is present.

### 1.10 Limitations and Known Issues

- **Relevance.** Retrieval quality has no absolute evaluation, because no relevance labels exist.
  `bge-small` is a general-purpose model, and the source claims are fragmented.
- **Ranking bias.** Best-chunk ranking slightly favours long patents, which have more chunks.
- **CPC parsing.** The main group and subgroup of codes with five or more digits are split
  heuristically (3 codes, flagged in the index).
- **Claim blocks.** Ambiguous claim blocks remain searchable, but their original claim boundaries are
  unknown.
- **Known bug: Unicode keyword matching.** A keyword filter for `strasse` matches a title containing
  `Straße` in the Python post-filter, because `casefold()` converts `ß` to `ss`. The FTS5 pre-filter
  (`unicode61` tokenizer) keeps the distinction, so changing the strategy can change which patents
  pass a keyword filter. The fix is to use the same tokenization in both paths and to add a
  regression test for this case.

## Part 2: Implementation at Scale

### 2.1 System Design

The complete design is in [docs/part2_system_design.md](docs/part2_system_design.md).

| Aspect | Design |
|---|---|
| Components | Amazon S3 (raw files, intermediate files, embeddings), Amazon RDS for PostgreSQL (records, text, job queue and batch status), Amazon OpenSearch Service (keyword and filter search; 378 million chunk vectors at one bit per dimension, re-scored at full precision), AWS Batch workers, and a search API on ECS Fargate |
| Ingestion pipeline | Six stages per weekly file: register, parse, embed, load, index, verify. A batch becomes searchable only after verification. |
| Query pipeline | Validate filters, obtain the query vector, resolve filters, run a filtered ANN search, group by patent, select evidence, assemble results. Expected latency is 40–180 ms. |
| Cost | USD 50–200 for the one-time backfill (about 110 GPU-hours) and USD 2,500–4,000 per month, mostly OpenSearch memory. A reduced configuration on PostgreSQL with pgvector costs about USD 1,000 per month but has lower recall. |
| Error handling | Retries with backoff and a heartbeat reaper for jobs; a `needs_review` state for bad batches; idempotent upserts and nightly reconciliation between the stores |
| Status tracking | `batches` and `jobs` tables, a pipeline dashboard, data-quality statistics per batch, latency and canary-recall monitoring |

The main decision is to index every chunk rather than one summary vector per patent. Summary vectors
were measured on the sample: re-scoring the top 20 candidates missed about 20% of the correct top 10,
and the top 100 (16% of the corpus) still missed 4–8%. The main challenges are:

- vector memory (1-bit compression and its recall loss);
- filtered search inside the ANN index;
- re-indexing when the model changes;
- the lack of relevance labels.

### 2.2 Proof of Concept

The proof of concept ([poc/README.md](poc/README.md)) implements the ingestion pipeline and job queue
of the design on one machine with Docker Compose.

| Container | Represents | Function |
|---|---|---|
| `db` (PostgreSQL 16) | RDS | Tables `batches`, `jobs`, `patents` and `chunks` (`poc/schema.sql`) |
| `scheduler` | EventBridge and stage 1 | Registers one weekly file per second, plus one corrupt file |
| `worker` (3 replicas) | AWS Batch | Claims jobs with `SELECT … FOR UPDATE SKIP LOCKED` and runs stages 2–6 with the Part 1 code |
| `dashboard` | CloudWatch | Batch states, stage progress, workers, errors and recoveries |

The workers inject faults:

- transient errors in 15% of jobs, which are retried with exponential backoff;
- crashes in 2% of jobs, which a reaper re-queues after 15 s without a heartbeat;
- one truncated file, which is moved to `needs_review` and not retried.

Results of one run:

- **Batches.** All 64 valid files were verified, and the corrupt file was held for review. The run
  took 97 s.
- **Recoveries.** 320 jobs completed, 71 transient errors were recovered by retry, and 10 jobs were
  recovered after worker crashes.
- **Data.** 636 patents and 20,872 chunks were loaded, the same as the Part 1 build with the same
  tokenizer.
- **Cross-batch deduplication.** The 4 A9 republications arrived eleven weeks after their originals.
  The load stage merged them across batches.

The proof of concept simplifies the design in four ways:

- a Docker volume replaces S3;
- a NumPy file per batch replaces OpenSearch;
- the embed stage uses the model-free `HashEmbedder`;
- every worker runs every stage.

## Part 3: Efficiency and Large Data

The full report is in [results/part3_efficiency.md](results/part3_efficiency.md).

### 3.1 Large Data Sample

`data/coding_challenge_data/` contains 471 files of 1,000 records each, in `job_000/` … `job_007/`.

| Property | Value | Consequence |
|---|---|---|
| Records | 471,000, of which 13,615 repeat a document number | Deduplication by document number is required before the database's unique key |
| Subject areas | All CPC sections, not only vehicles | The benchmark derives its filter scenarios from the indexed corpus |
| Classification | 38% `"N/A"` | These patents are indexed without a code; any CPC filter excludes them |
| Filenames | `US20240000148A1.XML` without a date; no `bibtex` field | Date filters are unavailable for this data |
| Text | 31.2 billion characters (about 7.8 billion tokens), 89% in descriptions | Embedding dominates the build time |

### 3.2 Method

| Item | Choice |
|---|---|
| Baseline | The Part 1 build and exact search, unchanged apart from the loader corrections in Section 3.6 |
| Test slice | The first 5 files of `job_000` (5,000 patents) for the baseline and every improvement; the first 10 files (10,000 patents) for the final configuration |
| Build measurements | Stage timings in each build's `manifest.json`; peak memory with `/usr/bin/time -l` |
| Search measurements | `patsearch bench` with `bench/queries_large.jsonl`; 100 warm runs per cell (median); recall@10 of the HNSW path against exact search |
| Quality measurements | `scripts/eval_agreement.py` against the baseline index. See the list below. |
| Acceptance rule | A lossless change must keep agreement@10 at about 1.0. A lossy change is accepted only with its measured cost. |

The quality measurements in `scripts/eval_agreement.py` are:

- **Query sets.** Claim 1 and the abstract of 200 random patents (each excluding its own patent), and
  the text queries of the benchmark.
- **Agreement@10.** The share of the baseline's top 10 patents that a changed configuration still
  returns.
- **Known-item test.** A description chunk from late in a patent must retrieve its own patent in the
  top 10 (recall@10 and MRR).

### 3.3 Ingestion: Lossless Improvements

`patsearch/fastbuild.py`, the default `build` profile, produces the same index layout as the Part 1
build.

| Change | Implementation | Measured effect |
|---|---|---|
| Float16 embedding | `model.half()` on MPS or CUDA. Rows with non-finite values are recomputed in float32, and vectors are normalised in float32 (`patsearch/embed.py`). | Embedding throughput 1.19 times higher on a 10,000-chunk sample (27,100 to 32,400 tokens/s). Cosine similarity with float32 vectors is 0.9999995. Batches above 64 were slower on MPS (128: 27,400; 256: 26,600 tokens/s), so the batch size stays 64. |
| Parallel CPU stage, overlapping with the GPU | A process pool (6 workers) loads, cleans, chunks and counts tokens for upcoming files while the main process embeds the current file. The separate token-check pass of the Part 1 build is removed. | Time outside embedding fell from 174 s to 45 s |
| Streaming, resumable build | Each input file becomes a shard on disk (records and float16 vectors), with a look-ahead of 2 files. A final pass assigns patent IDs in CPC order and writes SQLite and the memory-mapped matrix one shard at a time. A restarted build skips completed shards. | Peak memory fell from 5.81 GB to 2.27 GB. An interrupted build resumes without re-embedding; this was used once. |
| Deduplication before embedding | Repeated document numbers are removed first, then byte-identical records. Identical chunk texts within a file are embedded once. | 2,022 chunk texts (0.45%) and 1 repeated patent were skipped in the 5,000-patent slice; about 2.9% of the records in the full sample are duplicates |

| Configuration | Patents | Time | Patents/s | Tokens/s (embedding) | Peak memory | Agreement@10 with baseline |
|---|---|---|---|---|---|---|
| Baseline (`--profile baseline`) | 4,999 | 3,282 s | 1.52 | 26,400 | 5.81 GB | Reference |
| Fast build, lossless | 4,999 | 2,445 s | 2.04 | 34,200 | 2.27 GB | 0.995–1.0 |
| Fast build with `--max-desc-chunks 64` | 9,991 | 3,417 s | 2.92 | 32,200 | 2.43 GB | 0.88–0.94 |

Notes on the measurements:

- **Final build time.** It excludes 162 s during which the laptop was asleep (recorded by `pmset`).
- **Resumed build.** The final build was resumed once from its shards, so its `manifest.json` reports
  only the resumed run. The time above is taken from the log of the first run.
- **Small sample.** The fast build takes 99 s instead of 129 s and produces the same 636 patents and
  24,057 chunks.
- **Initial result.** Prefetching all prepared files filled memory and made the system swap. The
  fast build was then no faster than the baseline, until the look-ahead was limited to 2 files.

### 3.4 Ingestion: Description Limit

`--max-desc-chunks N` embeds the abstract, all claims and only the first N description chunks of each
patent. Its quality cost was measured without re-embedding, by removing the corresponding rows from
the baseline index. The known-item passages are chosen beyond the limit, so they are not indexed
themselves.

| N | Share of text embedded | Agreement@10 (claim 1 / abstract / text queries) | Known-item recall@10 |
|---|---|---|---|
| 8 | 22% | 0.67 / 0.62 / 0.59 | 0.90 |
| 16 | 32% | 0.73 / 0.69 / 0.67 | 0.89 |
| 32 | 50% | 0.81 / 0.79 / 0.82 | 0.92 |
| 64 (chosen) | 70% | 0.90 / 0.88 / 0.94 | 0.99 |
| All (baseline) | 100% | 1.0 | 1.0 |

N = 64 gives the largest saving with an agreement of about 0.9. Smaller values save more time but
change up to 40% of the results, because many matches come from late in the descriptions.

### 3.5 Search: HNSW Index

`patsearch ann-index` (`patsearch/ann.py`) builds a FAISS `IndexHNSWFlat` over the chunk vectors:

- inner-product metric;
- graph degree M = 32 and `efConstruction` = 80;
- the option `--sq8` stores 8-bit vectors in the graph instead.

`SearchEngine` loads the index if it exists. `AnnIndex` keeps the Part 1 cost-based logic:

- **Selective filters.** If a filter selects at most 100,000 rows, or a scoring path is forced, the
  exact path of Section 1.5 is used, and results are unchanged.
- **Broad filters or none.** The graph is searched for max(20 × k, 200) chunks, with
  `efSearch` set to at least that number (minimum 128). Filters, chunk kinds and excluded patents are applied inside the
  graph search through an `IDSelectorBitmap`, so the result is never under-filled as with a
  post-filter.
- **Re-scoring.** The candidate patents are re-scored exactly with their best chunk. Reported scores
  therefore equal exact scores, and the only possible loss is a missed candidate.

| Query | Exact, 445,000 vectors | HNSW, 445,000 vectors | Exact, 640,000 vectors | HNSW, 640,000 vectors |
|---|---|---|---|---|
| Unfiltered | 9.6 ms | 0.85 ms (recall@10 0.995) | 14.6 ms | 0.83 ms (recall@10 0.985) |
| Filtered (CPC, keyword, combination) | 0.05–1.2 ms | Unchanged (exact path) | 0.07–2.6 ms | Unchanged (exact path) |
| Naive post-filter, for comparison | 10–38 ms | | 15–50 ms | |
| Claim mapping, end to end | 10.6 ms | 2.0 ms | | 2.4 ms |

- **Build cost.** Building the index takes 18 s for 445,000 vectors. It adds about 0.8 GB (5,000
  patents) and 1.2 GB (10,000 patents) on disk.
- **Text queries.** Their time is dominated by query embedding (about 6.3 ms).
- **Tests.** `tests/test_ann.py` checks that HNSW results with and without filters equal exact
  results on a small corpus.

### 3.6 Problems Found at Scale

| Problem | Correction |
|---|---|
| Filenames have no date; files are nested in `job_*/` directories | The date is optional in the filename parser, and `list_data_files` finds the nested files (`patsearch/load.py`) |
| 13,615 repeated document numbers would violate the database's unique key | Deduplication by document number before deduplication by content; counted in the load statistics |
| A claim list that begins with unnumbered text produced two units numbered "claim 1", which stopped the final build on the claims table's primary key. The error was latent in Part 1. | Leading unnumbered text now joins the first numbered claim (`patsearch/claims.py`, with a regression test). The final pass also merges duplicate claim numbers. The build was resumed from its shards without re-embedding. |
| PyTorch and FAISS each include an OpenMP runtime on macOS; whichever starts second stops the process ("OMP: Error #15"), whatever the import order | `KMP_DUPLICATE_LIB_OK=TRUE` is set in `patsearch/__init__.py` before either library loads. Correctness is checked by `tests/test_ann.py` and by the recall measurements. |
| Unbounded prefetching filled memory | The look-ahead is limited to 2 files |

### 3.7 Extrapolation to the Full Sample

| Configuration | Time on this laptop | Chunk vectors (float32) |
|---|---|---|
| Baseline | About 86 h | About 42 million (64 GB) |
| Fast build, lossless | About 64 h | About 42 million (64 GB) |
| Final configuration | About 45 h | About 30 million (46 GB) |

One laptop is not sufficient for the full sample, because the vectors exceed both memory and free
disk space. The next steps follow the Part 2 design:

- embedding on several GPUs (about 110 T4 GPU-hours for the full sample);
- float16 or 8-bit vectors on disk (`ann-index --sq8` already stores 8-bit vectors in the graph);
- a compressed index (IVF-PQ) or a search service such as OpenSearch.

### 3.8 Result Files

| File | Content | Produced by |
|---|---|---|
| `results/part3_efficiency.md` | Part 3 report | Written from the files below |
| `results/part3_baseline_search.md`, `.json` | Exact search on the 5,000-patent baseline | `patsearch bench` before `ann-index` |
| `results/part3_ann_search.md`, `.json` | HNSW search on the same index | `patsearch bench` after `ann-index` |
| `results/part3_final_search.md`, `.json` | HNSW search on the 10,000-patent final index | `patsearch bench` |
| `results/part3_quality_fast_vs_baseline.json` | Agreement of the fast build with the baseline | `scripts/eval_agreement.py --cand` |
| `results/part3_quality_truncation.json` | Description-limit sweep | `scripts/eval_agreement.py --desc-chunks` |
| `results/part3_embed_microbench.json` | Float16 and batch-size measurements | `scripts/bench_embed.py` |
| `results/bench_part1*.md`, `.json` | Part 1 benchmark (default and single BLAS thread) | `patsearch bench --synthetic 100000,1000000` |

The commands that produce these files are listed in the [README](README.md#part-3).
