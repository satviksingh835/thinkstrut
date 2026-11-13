# Part 3: Efficiency and Large Data

## Problem

The large sample (`data/coding_challenge_data/`) contains 471 files of 1,000 patents (471,000
records): all CPC sections, 13,615 duplicate records of the same doc_number, 38% of classifications
`"N/A"`, no publication dates, and 31.2 billion characters of text (about 7.8 billion tokens, 89% of it
in descriptions). At the Part 1 rate, embedding all of it would take about 86 hours on the test machine.
The goal was to measure ingestion and search on a portion of the data, improve both, and keep the
fastest code.

**Setup.** Apple M4 (10 cores, 16 GB, MPS GPU), macOS, Python 3.12. Baseline and every improvement
were measured on the same slice: the first 5 files of `job_000` (5,000 patents). The final
configuration was run on the first 10 files (10,000 patents). Build timings come from each build's
`manifest.json` and `/usr/bin/time -l`; search timings from `patsearch bench` (100 warm repetitions,
median); quality from `scripts/eval_agreement.py`.

## Results summary

| | Baseline (Part 1) | Final configuration |
|---|---|---|
| Ingestion rate | 1.52 patents/s | **2.92 patents/s (1.93×)** |
| Peak memory during the build | 5.81 GB (5k patents) | **2.43 GB (10k patents)** |
| Unfiltered search | 9.6 ms (445k vectors, exact) | **0.83 ms (640k vectors, HNSW)** |
| Quality relative to baseline | reference | lossless steps: agreement@10 0.995–1.0; truncation: 0.88–0.94; ANN recall@10: 0.985 |

## 1. Ingestion

| Configuration | Patents | Time | Patents/s | Tokens/s (embedding) | Peak memory | Agreement@10 with baseline |
|---|---|---|---|---|---|---|
| Baseline: Part 1 build (`--profile baseline`) | 4,999 | 3,282 s | 1.52 | 26.4k | 5.81 GB | reference |
| Fast build, lossless (`--profile fast`) | 4,999 | 2,445 s | 2.04 | 34.2k | 2.27 GB | 0.995–1.0 |
| Fast build + first 64 description chunks | 9,991 | 3,417 s¹ | **2.92** | 32.2k | 2.43 GB | 0.88–0.94 |

¹ 3,574 s wall-clock minus 162 s during which the laptop was asleep (lid closed; `pmset` log).

**Lossless improvements** (`patsearch/fastbuild.py`; same index format, no change in results):

| Change | Measured effect |
|---|---|
| Embed in float16 on the GPU (`--dtype float16`) | 1.19× embedding throughput on a 10,000-chunk sample (27.1k to 32.4k tokens/s); cosine with float32 vectors 0.9999995. Batch sizes above 64 were slower on MPS (128: 27.4k, 256: 26.6k tokens/s) |
| Remove the second tokenisation pass; run loading, cleaning and chunking in a process pool that overlaps with GPU embedding | Non-embedding time 174 s to 45 s |
| Streaming build: one shard per input file, bounded look-ahead, final assembly shard by shard; resumable | Peak memory 5.81 GB to 2.27 GB; an interrupted build resumes without re-embedding (used once in this run) |
| Deduplicate before embedding (repeated doc_numbers, identical chunk texts within a file) | 2,022 chunk texts (0.45%) and 1 repeated patent skipped in the 5k slice; about 2.9% of records in the full dataset |

**Lossy improvement: embed only the first N description chunks** (`--max-desc-chunks N`). The quality
cost was measured without re-embedding, by removing rows from the baseline index (`results/part3_quality_truncation.json`).
Agreement@10 is the share of the baseline top 10 that remains; the known-item test checks that a
passage from late in a description still retrieves its own patent when that passage is not indexed.

| N | Share of text embedded | Agreement@10 (claim 1 / abstract / text queries) | Known-item recall@10 |
|---|---|---|---|
| 8 | 22% | 0.67 / 0.62 / 0.59 | 0.90 |
| 16 | 32% | 0.73 / 0.69 / 0.67 | 0.89 |
| 32 | 50% | 0.81 / 0.79 / 0.82 | 0.92 |
| **64 (chosen)** | **70%** | **0.90 / 0.88 / 0.94** | **0.99** |
| all (baseline) | 100% | 1.0 | 1.0 |

N = 64 was chosen as the largest saving with agreement of about 0.9. Truncating further saves more time
but changes up to 40% of results, because many matches come from deep in the descriptions.

## 2. Search

`patsearch ann-index` builds a FAISS HNSW index (M = 32) over the chunk vectors (`patsearch/ann.py`).
Broad or unfiltered queries use the graph, with filters applied inside the graph search through a row
bitmap; the candidate patents are then re-scored exactly. Selective filters (up to 100,000 rows) keep
the exact path, so their results are unchanged.

| Query | Exact, 445k vectors | HNSW, 445k vectors | Exact, 640k vectors | HNSW, 640k vectors |
|---|---|---|---|---|
| Unfiltered | 9.6 ms | **0.85 ms** (recall@10 0.995) | 14.6 ms | **0.83 ms** (recall@10 0.985) |
| Filtered (CPC, keyword, combination) | 0.05–1.2 ms | unchanged (exact path) | 0.07–2.6 ms | unchanged (exact path) |
| Naive post-filter (for comparison) | 10–38 ms | | 15–50 ms | |
| Claim mapping, end to end | 10.6 ms | **2.0 ms** | | **2.4 ms** |

The HNSW index takes 18 s to build for 445k vectors and adds about 0.8 GB (5k) and 1.2 GB (10k) to the
index on disk. Text queries are dominated by query embedding (about 6.3 ms on the GPU).

## 3. Problems found at scale

| Problem | Fix |
|---|---|
| Filenames in the large dataset have no date | Date made optional in the filename parser; date filters unavailable for this data |
| 13,615 repeated doc_numbers would violate the database's unique key | Deduplication by doc_number before content hashing |
| A patent whose claim list starts with unnumbered text produced two units numbered "claim 1" (latent Part 1 bug) | Fixed in `claims.py`, with a regression test; final assembly also merges duplicate claim numbers |
| torch and faiss each ship an OpenMP runtime; whichever starts second aborts the process | Duplicate runtime allowed before either library loads; ANN results verified identical to exact search in tests |
| Prefetching all prepared files filled memory; with other applications running, the machine swapped and the float16 gain disappeared | Look-ahead limited to 2 files |

## 4. Extrapolation to all 471,000 patents

| Configuration | Time on this laptop | Chunk vectors (float32) |
|---|---|---|
| Baseline | ~86 h | ~42 million (64 GB) |
| Fast build, lossless | ~64 h | ~42 million (64 GB) |
| Final configuration | ~45 h | ~30 million (46 GB) |

A single laptop is not sufficient for the full set: the vectors exceed both memory and free disk space.
The next steps are those of the Part 2 design: embedding on several GPUs (about 110 T4 GPU-hours for the
full set), float16 or 8-bit vectors on disk (`patsearch ann-index --sq8` already stores 8-bit vectors in
the graph), and a compressed index (IVF-PQ) or a search service such as OpenSearch.

## Reproducing

```bash
uv run patsearch build --profile baseline --data-dir data/coding_challenge_data --limit-files 5 --out artifacts/large_5k_baseline
uv run patsearch build --profile fast     --data-dir data/coding_challenge_data --limit-files 5 --out artifacts/large_5k_fast
uv run patsearch build --profile fast --max-desc-chunks 64 --data-dir data/coding_challenge_data --limit-files 10 --out artifacts/large_10k_final
uv run patsearch ann-index --index artifacts/large_10k_final
uv run python scripts/make_large_queries.py --index artifacts/large_5k_baseline
uv run patsearch bench --index artifacts/large_10k_final --queries bench/queries_large.jsonl --reps 100 --out results/part3_final_search.md
uv run python scripts/bench_embed.py --index artifacts/large_5k_baseline
uv run python scripts/eval_agreement.py --ref artifacts/large_5k_baseline --desc-chunks 8,16,32,64
uv run python scripts/eval_agreement.py --ref artifacts/large_5k_baseline --cand artifacts/large_5k_fast
```
