# Part 1 benchmark — search latency with vs without hybrid filters

- machine: Apple M4 · Darwin arm64 · numpy 2.5.4
- index: build `56e00f0fca2d`, 4999 patents, 444,820 chunk vectors × 384 dims (float32, 683 MB), model `BAAI/bge-small-en-v1.5`
- BLAS threads: VECLIB_MAXIMUM_THREADS=default; reps per cell 100 (after 5 warm-up calls), k=10; times are wall-clock `perf_counter_ns`

## Query embedding (model inference, independent of filters)

p50 **6.18 ms**, p95 6.8 ms on `mps` (text queries; patent→similar and claim-by-reference reuse stored vectors and need no model call).

## Retrieval on the real index (filter + score + aggregate + top-k), text & claim query vectors

| scenario | docs kept | rows kept | strategy | path | p50 ms | p95 ms | filter p50 | score p50 | post-filter p50 | recall@10 vs pre | same top-10 as pre |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 4999 | 100.0% | pre (auto path) | hnsw | 0.846 | 2.767 | 0.0 | 0.224 |  | 0.995 | False |
| cpc=G06F | 260 | 5.2% | pre (auto path) | slices | 0.587 | 0.611 | 0.067 | 0.506 |  | 1.0 | True |
| cpc=G06F | 260 | 5.2% | pre, force gather | gather | 1.874 | 2.267 | 0.085 | 1.755 |  | 1.0 | True |
| cpc=G06F | 260 | 5.2% | pre, force full+mask | full | 9.587 | 10.297 | 0.099 | 9.456 |  | 1.0 | True |
| cpc=G06F | 260 | 5.2% | post (exhaustive) | full | 9.873 | 10.613 |  | 9.42 | 0.103 | 1.0 | True |
| cpc=G06F | 260 | 5.2% | post-top100 | full+top100 | 9.648 | 11.665 |  | 9.446 | 0.055 | 0.515 | False |
| cpc=H01M | 51 | 1.1% | pre (auto path) | slices | 0.052 | 0.062 | 0.017 | 0.028 |  | 1.0 | True |
| cpc=H01M | 51 | 1.1% | pre, force gather | gather | 0.222 | 0.251 | 0.018 | 0.197 |  | 1.0 | True |
| cpc=H01M | 51 | 1.1% | pre, force full+mask | full | 9.497 | 10.287 | 0.05 | 9.423 |  | 1.0 | True |
| cpc=H01M | 51 | 1.1% | post (exhaustive) | full | 10.712 | 11.632 |  | 9.444 | 0.826 | 1.0 | True |
| cpc=H01M | 51 | 1.1% | post-top100 | full+top100 | 9.623 | 10.425 |  | 9.438 | 0.053 | 0.12 | False |
| kw=vehicle | 238 | 3.7% | pre (auto path) | gather | 1.228 | 1.391 | 0.118 | 1.091 |  | 1.0 | True |
| kw=vehicle | 238 | 3.7% | pre, force gather | gather | 1.216 | 1.309 | 0.116 | 1.085 |  | 1.0 | True |
| kw=vehicle | 238 | 3.7% | pre, force full+mask | full | 9.659 | 10.479 | 0.166 | 9.454 |  | 1.0 | True |
| kw=vehicle | 238 | 3.7% | post (exhaustive) | full | 38.235 | 89.014 |  | 9.426 | 28.45 | 1.0 | True |
| kw=vehicle | 238 | 3.7% | post-top100 | full+top100 | 17.371 | 18.184 |  | 9.418 | 7.705 | 0.23 | False |
| cpc=G06F & kw=signal | 14 | 0.2% | pre (auto path) | slices | 0.121 | 0.128 | 0.082 | 0.032 |  | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | pre, force gather | gather | 0.138 | 0.147 | 0.083 | 0.048 |  | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | pre, force full+mask | full | 9.632 | 10.374 | 0.184 | 9.432 |  | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | post (exhaustive) | full | 26.337 | 31.973 |  | 9.427 | 16.441 | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | post-top100 | full+top100 | 10.086 | 11.539 |  | 9.433 | 0.409 | 0.045 | False |

## Patent → similar (stage 1 doc vectors, stage 2 max-sim over 50 candidates), no hydration

| scenario | p50 ms | p95 ms | stage 1 p50 | stage 2 p50 |
|---|---|---|---|---|
| none | 1.164 | 3.325 | 0.265 | 0.877 |
| cpc=G06F | 0.972 | 1.38 | 0.026 | 0.887 |
| cpc=H01M | 0.913 | 1.314 | 0.016 | 0.879 |
| kw=vehicle | 0.923 | 1.404 | 0.026 | 0.807 |

## End-to-end per call (model loaded; includes query embedding and SQLite hydration of results + evidence)

| input | p50 ms | p95 ms | hydrate p50 |
|---|---|---|---|
| text | 7.72 | 8.29 | 0.33 |
| claim (by ref) | 2.03 | 3.15 | 0.32 |
| similar | 1.28 | 3.47 | 0.14 |
