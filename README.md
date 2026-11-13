# Patent Search Engine

`patsearch` is a hybrid semantic search engine for US patent applications. It ranks patents by the meaning of a query and returns the claim or
paragraph that matched. Searches can be restricted by classification code, keyword, title and date.
Part 3 extends the engine to the large data sample.

| Document | Content |
|---|---|
| [README.md](README.md) | Problem statement, approach, and instructions to reproduce the results |
| [TECHNICAL.md](TECHNICAL.md) | Technical details for each part: data handling, design, and measurements |
| [docs/part2_system_design.md](docs/part2_system_design.md) | Part 2: system design for 10 million patents |
| [poc/README.md](poc/README.md) | Part 2: proof of concept |
| [TECHNICAL.md, Part 3](TECHNICAL.md#part-3-efficiency-and-large-data) | Part 3: efficiency measurements and results |

Video walkthrough of Parts 2 and 3: *link to be added*.

## Problem Statement

A patent examiner reviewing a claim must find the prior art that overlaps with it: claims and
description paragraphs in other patents that describe the same invention. The search is usually
limited to one technology area or to documents that contain known terms. This project addresses
claim-level semantic search with hybrid constraints. It must stay fast as the corpus grows from the
sample's 10^3 patents towards 10^7.

The engine accepts three kinds of input:

1. A natural-language description of an invention.
2. A patent number, to find similar patents.
3. A claim, by reference (`20240051333:1`) or as text, to find the closest claims and paragraphs in
   other patents (claim mapping).

Each result is a patent with its metadata and the passages that explain the match (claims,
description paragraphs or the abstract). Any input can be combined with constraints on the CPC
classification code, keywords in the title or abstract, the exact title, and the publication date.

## Approach

### Part 1: Search Engine

- **Chunking and embedding.** Each patent is split into chunks: the abstract, each claim, and
  description paragraphs of about 250 tokens. Each chunk is embedded with `BAAI/bge-small-en-v1.5`.
  A patent's score is the score of its best chunk, so a match in a single claim is not diluted by the
  rest of the document.
- **Storage.** Metadata, text and a full-text keyword index (FTS5) are stored in SQLite. The vectors
  are stored in a memory-mapped NumPy matrix.
- **Hybrid search.** Constraints are resolved first through indexes, and only the vectors of the
  matching patents are scored. Patents are stored in order of CPC subclass and publication date, so
  most filters select contiguous ranges of the matrix.

**Missing and malformed data.** No patent is excluded. The main decisions are listed below; all
findings are listed in [TECHNICAL.md](TECHNICAL.md#11-data-analysis-and-handling).

| Finding | Decision |
|---|---|
| 119 of the 640 patents have an empty detailed description | Kept and searchable through the title, abstract and claims, and marked as having no description |
| Claims are stored as fragments; nested claims lost their numbers and canceled claims remain in the list | Claims are re-assembled by number, and canceled claims are removed |
| 4 corrected republications (kind code A9) are identical to their originals | Indexed once under the original (A1) number; the A9 number remains a valid alias |
| Absent, null or wrongly typed fields (none in the sample; present in the large data) | Treated as empty; a record is dropped only if it has no document number or no text |

**Timing with and without hybrid search.** The table shows retrieval time on the sample index (636
patents, 24,057 vectors). Each value is the median of 200 runs on an Apple M4. Query embedding
(6.0 ms) is excluded, because it does not depend on the filters.

| Constraint | Patents kept | Filter first (this engine) | Naive post-filter |
|---|---|---|---|
| None | 100% | 0.53 ms | |
| CPC `B60C` | 48.5% | 0.29 ms | 0.60 ms |
| CPC `B60D` | 3.3% | 0.033 ms | 0.76 ms |
| Keyword `bicycle` | 6.0% | 0.074 ms | 21.9 ms |
| CPC `B60B` and keyword `bicycle` | 5.8% | 0.081 ms | 6.4 ms |

With filters, search is faster than without them. The constraints are resolved through indexes in
less than 0.1 ms. Vector scoring, which dominates the cost, then covers only the selected rows. On a
synthetic index of one million vectors, an unfiltered search takes 21.9 ms and a 2.9% CPC filter
0.66 ms. Filtering the top 100 results of an approximate search instead loses relevant patents when
the filter is selective: recall@10 is 0.14 for `B60D`. The full measurements and commentary are in
[TECHNICAL.md, Part 1](TECHNICAL.md#16-timing-with-and-without-hybrid-search).

### Part 2: Implementation at Scale

- **System design** ([docs/part2_system_design.md](docs/part2_system_design.md)). The design uses
  Amazon S3, PostgreSQL, and OpenSearch holding 378 million chunk vectors compressed to one bit per
  dimension. A six-stage batch pipeline ingests one weekly file at a time.
  - Estimated cost: USD 50–200 for the one-time backfill and USD 2,500–4,000 per month.
  - The design indexes every chunk rather than one vector per patent. In a measurement on the sample,
    per-patent vectors missed 4–8% of the correct top 10 even when 100 candidates were re-scored.
- **Proof of concept** ([poc/](poc/README.md)). The ingestion pipeline runs in Docker: a PostgreSQL
  job queue, a scheduler, three workers and a status dashboard. With injected faults, it processed all
  64 files, recovered from 71 transient errors and 10 worker crashes, and held back a corrupt file for
  review. It loaded the same 636 patents as Part 1.

### Part 3: Enhancement: Efficiency and Large Data

**Why this enhancement.** At the Part 1 rate, embedding the 471,000-patent large sample would take
about 86 hours on the test machine. Exact search time also grows linearly with the number of vectors.
The Part 2 design showed that embedding and vector search set the cost and latency at 10^7 patents.
Measuring and reducing both on real data is therefore the most direct test of whether the Part 1
design can scale.

**Method.** The baseline and each improvement were measured on the same 5,000-patent slice of the
large sample. The final configuration was run on 10,000 patents. A change that alters results was
accepted only when its quality cost had been measured. Quality was measured as agreement@10: the
share of the baseline's top 10 patents that a changed configuration still returns.

| Measure | Baseline (Part 1 code) | Final configuration |
|---|---|---|
| Ingestion rate | 1.52 patents/s | 2.92 patents/s (1.93×) |
| Peak memory during the build | 5.81 GB (5,000 patents) | 2.43 GB (10,000 patents) |
| Unfiltered search | 9.6 ms (445,000 vectors, exact) | 0.83 ms (640,000 vectors, recall@10 0.985) |
| Claim mapping, end to end | 10.6 ms | 2.4 ms |

1. **Streaming build** (`patsearch build`, the default `fast` profile). The build changes are lossless
   (agreement@10 with the baseline 0.995–1.0):
   - parsing and chunking run in a process pool while the GPU embeds;
   - embedding runs in float16;
   - duplicates are removed before embedding;
   - every input file becomes a resumable shard on disk.
2. **Description limit** (`--max-desc-chunks 64`). Only the first 64 description chunks of each patent
   are embedded, which is 70% of the text. Agreement@10 is 0.88–0.94, and a passage from late in a
   description still retrieves its own patent in 99% of cases.
3. **HNSW index** (`patsearch ann-index`). FAISS graph search handles unfiltered and broad queries.
   Filters are applied inside the graph search, and candidates are re-scored exactly. Selective
   filters keep the exact path, so their results are unchanged.
4. **Corrections required by the large data.** Filenames without dates, 13,615 duplicate document
   numbers, a claim-numbering error, and a conflict between the OpenMP runtimes of PyTorch and FAISS.

## How to Run

Requirements:

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An Apple Silicon (MPS) or CUDA GPU is recommended. The code also runs on a CPU, but builds are
  slower.
- The first build downloads the embedding model (about 130 MB).

```bash
uv sync
```

Unzip the data into `data/` (the directory is not tracked by git):

| Dataset | Location |
|---|---|
| Small sample (64 files `patents_ipa*.json`) | `data/patent_data_small/` |
| Large sample (`job_000/` … `job_007/`, files `patents_batch_*.json`) | `data/coding_challenge_data/` |

### Part 1

Build the index (about 2 minutes on an Apple M4). Each build is written to `artifacts/<build_id>/`,
and later commands use the most recent build.

```bash
uv run patsearch build --profile baseline
```

`--profile baseline` runs the Part 1 build. Without it, `build` runs the Part 3 build, which produces
the same patents and chunks with float16 embeddings.

Search:

```bash
uv run patsearch search "tire with sound absorbing foam in the cavity to reduce noise"
uv run patsearch search "quick release for the wheel axle" --cpc B60B --kw bicycle
uv run patsearch search "noise" --title "pneumatic tire" --date-from 2025-01-01
uv run patsearch similar 20240131861                 # similar patents
uv run patsearch claim 20240051333:1                 # claim mapping from a stored claim
uv run patsearch claim --text "a spoke made of carbon fiber with threaded ends" --granularity passage
uv run patsearch show 20240051338                    # metadata and re-assembled claims
uv run patsearch serve                               # web interface at http://127.0.0.1:8000
```

Reproduce the timing results (written to `results/bench_part1.md`):

```bash
uv run patsearch bench --synthetic 100000,1000000
```

Run the tests. They use a synthetic corpus and do not download the model.

```bash
uv run pytest
```

The filter options are:

- `--cpc` (repeatable, any match);
- `--kw` (repeatable, all must match; `--kw-any` for any);
- `--kw-field title|abstract|both`;
- `--title` (exact title) and `--title-contains`;
- `--date-from` and `--date-to`.

`uv run patsearch <command> --help` lists all options.

### Part 3

The commands below reproduce every Part 3 result in
[TECHNICAL.md](TECHNICAL.md#part-3-efficiency-and-large-data). The builds take 40–60 minutes each on
an Apple M4. To also record peak memory, prefix a build with `/usr/bin/time -l`.

```bash
# 1. Baseline: Part 1 build on the first 5 files (5,000 patents), and its search benchmark
uv run patsearch build --profile baseline --data-dir data/coding_challenge_data --limit-files 5 --out artifacts/large_5k_baseline
uv run patsearch bench --index artifacts/large_5k_baseline --queries bench/queries_large.jsonl --reps 100 --out results/part3_baseline_search.md

# 2. Lossless fast build on the same files, and its agreement with the baseline
uv run patsearch build --data-dir data/coding_challenge_data --limit-files 5 --out artifacts/large_5k_fast
uv run python scripts/eval_agreement.py --ref artifacts/large_5k_baseline --cand artifacts/large_5k_fast --out results/part3_quality_fast_vs_baseline.json

# 3. Quality cost of the description limit (simulated on the baseline), and embedding precision
uv run python scripts/eval_agreement.py --ref artifacts/large_5k_baseline --desc-chunks 8,16,32,64 --out results/part3_quality_truncation.json
uv run python scripts/bench_embed.py --index artifacts/large_5k_baseline

# 4. HNSW index on the baseline, and the same benchmark again
uv run patsearch ann-index --index artifacts/large_5k_baseline
uv run patsearch bench --index artifacts/large_5k_baseline --queries bench/queries_large.jsonl --reps 100 --out results/part3_ann_search.md

# 5. Final configuration on the first 10 files (10,000 patents)
uv run patsearch build --max-desc-chunks 64 --data-dir data/coding_challenge_data --limit-files 10 --out artifacts/large_10k_final
uv run patsearch ann-index --index artifacts/large_10k_final
uv run patsearch bench --index artifacts/large_10k_final --queries bench/queries_large.jsonl --reps 100 --out results/part3_final_search.md
uv run patsearch search "lithium battery cathode material" --index artifacts/large_10k_final
```

Stage timings are printed at the end of each build and saved in its `manifest.json`. The benchmark
queries in `bench/queries_large.jsonl` are committed. They were generated with
`uv run python scripts/make_large_queries.py --index artifacts/large_5k_baseline`.

### Part 2 Proof of Concept

Docker must be running, and the small sample must be in `data/patent_data_small/`.

```bash
docker compose -f poc/docker-compose.yml up --build -d    # dashboard at http://localhost:8080
docker compose -f poc/docker-compose.yml down -v          # stop and delete all data
```

## Repository Layout

```text
patsearch/   search engine: data loading, index build, search, CLI, HTTP API and web interface
tests/       unit and engine tests on a synthetic corpus
scripts/     data profiling and Part 3 measurement scripts
bench/       benchmark queries for the sample and the large data
results/     benchmark reports and Part 3 measurements
docs/        Part 2 system design and architecture diagram
poc/         Part 2 proof of concept
```

## Limitations

- **Retrieval quality.** It was measured only relative to the baseline (agreement@10). No relevance
  labels were available for an absolute evaluation.
- **Large sample.** It was indexed only in part (10,000 of 471,000 patents). Indexing all of it
  requires the distributed design of Part 2.
- **Further limitations and known issues.** They are listed in
  [TECHNICAL.md](TECHNICAL.md#110-limitations-and-known-issues).
