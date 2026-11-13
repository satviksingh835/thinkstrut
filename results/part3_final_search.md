# Part 1 benchmark — search latency with vs without hybrid filters

- machine: Apple M4 · Darwin arm64 · numpy 2.5.4
- index: build `f7941125bab6`, 9991 patents, 640,332 chunk vectors × 384 dims (float32, 984 MB), model `BAAI/bge-small-en-v1.5`
- BLAS threads: VECLIB_MAXIMUM_THREADS=default; reps per cell 100 (after 5 warm-up calls), k=10; times are wall-clock `perf_counter_ns`

## Query embedding (model inference, independent of filters)

p50 **6.32 ms**, p95 8.14 ms on `mps` (text queries; patent→similar and claim-by-reference reuse stored vectors and need no model call).

## Retrieval on the real index (filter + score + aggregate + top-k), text & claim query vectors

| scenario | docs kept | rows kept | strategy | path | p50 ms | p95 ms | filter p50 | score p50 | post-filter p50 | recall@10 vs pre | same top-10 as pre |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 9991 | 100.0% | pre (auto path) | hnsw | 0.833 | 1.929 | 0.0 | 0.191 |  | 0.985 | False |
| cpc=G06F | 487 | 5.2% | pre (auto path) | slices | 0.86 | 0.931 | 0.114 | 0.727 |  | 1.0 | True |
| cpc=G06F | 487 | 5.2% | pre, force gather | gather | 2.642 | 2.96 | 0.132 | 2.478 |  | 1.0 | True |
| cpc=G06F | 487 | 5.2% | pre, force full+mask | full | 14.605 | 21.852 | 0.168 | 14.387 |  | 1.0 | True |
| cpc=G06F | 487 | 5.2% | post (exhaustive) | full | 15.252 | 22.896 |  | 14.27 | 0.139 | 1.0 | True |
| cpc=G06F | 487 | 5.2% | post-top100 | full+top100 | 14.245 | 16.059 |  | 13.902 | 0.058 | 0.475 | False |
| cpc=H01M | 95 | 1.0% | pre (auto path) | slices | 0.071 | 0.08 | 0.025 | 0.038 |  | 1.0 | True |
| cpc=H01M | 95 | 1.0% | pre, force gather | gather | 0.278 | 0.292 | 0.025 | 0.244 |  | 1.0 | True |
| cpc=H01M | 95 | 1.0% | pre, force full+mask | full | 14.466 | 16.441 | 0.08 | 14.345 |  | 1.0 | True |
| cpc=H01M | 95 | 1.0% | post (exhaustive) | full | 15.876 | 16.948 |  | 13.58 | 1.573 | 1.0 | True |
| cpc=H01M | 95 | 1.0% | post-top100 | full+top100 | 13.956 | 15.991 |  | 13.635 | 0.056 | 0.11 | False |
| kw=vehicle | 452 | 4.0% | pre (auto path) | gather | 2.567 | 3.242 | 0.283 | 2.241 |  | 1.0 | True |
| kw=vehicle | 452 | 4.0% | pre, force gather | gather | 2.351 | 3.002 | 0.244 | 2.029 |  | 1.0 | True |
| kw=vehicle | 452 | 4.0% | pre, force full+mask | full | 14.674 | 16.225 | 0.296 | 14.335 |  | 1.0 | True |
| kw=vehicle | 452 | 4.0% | post (exhaustive) | full | 49.568 | 141.484 |  | 13.641 | 34.598 | 1.0 | True |
| kw=vehicle | 452 | 4.0% | post-top100 | full+top100 | 21.847 | 23.574 |  | 13.681 | 7.786 | 0.19 | False |
| cpc=G06F & kw=signal | 28 | 0.3% | pre (auto path) | slices | 0.491 | 0.511 | 0.431 | 0.052 |  | 1.0 | True |
| cpc=G06F & kw=signal | 28 | 0.3% | pre, force gather | gather | 0.521 | 0.533 | 0.432 | 0.08 |  | 1.0 | True |
| cpc=G06F & kw=signal | 28 | 0.3% | pre, force full+mask | full | 14.624 | 16.745 | 0.77 | 13.832 |  | 1.0 | True |
| cpc=G06F & kw=signal | 28 | 0.3% | post (exhaustive) | full | 33.595 | 39.645 |  | 13.605 | 19.379 | 1.0 | True |
| cpc=G06F & kw=signal | 28 | 0.3% | post-top100 | full+top100 | 14.443 | 16.112 |  | 13.598 | 0.397 | 0.03 | False |

## Patent → similar (stage 1 doc vectors, stage 2 max-sim over 50 candidates), no hydration

| scenario | p50 ms | p95 ms | stage 1 p50 | stage 2 p50 |
|---|---|---|---|---|
| none | 1.434 | 2.059 | 0.614 | 0.821 |
| cpc=G06F | 0.942 | 1.307 | 0.036 | 0.802 |
| cpc=H01M | 0.841 | 1.199 | 0.019 | 0.792 |
| kw=vehicle | 0.966 | 1.364 | 0.039 | 0.77 |

## End-to-end per call (model loaded; includes query embedding and SQLite hydration of results + evidence)

| input | p50 ms | p95 ms | hydrate p50 |
|---|---|---|---|
| text | 8.05 | 10.54 | 0.3 |
| claim (by ref) | 2.36 | 5.65 | 0.31 |
| similar | 1.78 | 5.05 | 0.18 |
