# Part 1 benchmark — search latency with vs without hybrid filters

- machine: Apple M4 · Darwin arm64 · numpy 2.5.4
- index: build `56e00f0fca2d`, 4999 patents, 444,820 chunk vectors × 384 dims (float32, 683 MB), model `BAAI/bge-small-en-v1.5`
- BLAS threads: VECLIB_MAXIMUM_THREADS=default; reps per cell 100 (after 5 warm-up calls), k=10; times are wall-clock `perf_counter_ns`

## Query embedding (model inference, independent of filters)

p50 **6.54 ms**, p95 7.13 ms on `mps` (text queries; patent→similar and claim-by-reference reuse stored vectors and need no model call).

## Retrieval on the real index (filter + score + aggregate + top-k), text & claim query vectors

| scenario | docs kept | rows kept | strategy | path | p50 ms | p95 ms | filter p50 | score p50 | post-filter p50 | recall@10 vs pre | same top-10 as pre |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 4999 | 100.0% | pre (auto path) | full | 9.628 | 10.492 | 0.001 | 9.478 |  | 1.0 | True |
| cpc=G06F | 260 | 5.2% | pre (auto path) | slices | 0.598 | 0.666 | 0.072 | 0.509 |  | 1.0 | True |
| cpc=G06F | 260 | 5.2% | pre, force gather | gather | 1.682 | 2.018 | 0.076 | 1.588 |  | 1.0 | True |
| cpc=G06F | 260 | 5.2% | pre, force full+mask | full | 9.629 | 10.263 | 0.102 | 9.492 |  | 1.0 | True |
| cpc=G06F | 260 | 5.2% | post (exhaustive) | full | 9.918 | 10.647 |  | 9.458 | 0.103 | 1.0 | True |
| cpc=G06F | 260 | 5.2% | post-top100 | full+top100 | 10.521 | 15.515 |  | 10.261 | 0.063 | 0.515 | False |
| cpc=H01M | 51 | 1.1% | pre (auto path) | slices | 0.049 | 0.052 | 0.016 | 0.026 |  | 1.0 | True |
| cpc=H01M | 51 | 1.1% | pre, force gather | gather | 0.215 | 0.261 | 0.017 | 0.191 |  | 1.0 | True |
| cpc=H01M | 51 | 1.1% | pre, force full+mask | full | 9.588 | 10.176 | 0.054 | 9.501 |  | 1.0 | True |
| cpc=H01M | 51 | 1.1% | post (exhaustive) | full | 10.872 | 11.794 |  | 9.477 | 0.873 | 1.0 | True |
| cpc=H01M | 51 | 1.1% | post-top100 | full+top100 | 9.62 | 10.168 |  | 9.424 | 0.054 | 0.12 | False |
| kw=vehicle | 238 | 3.7% | pre (auto path) | gather | 1.233 | 1.465 | 0.122 | 1.094 |  | 1.0 | True |
| kw=vehicle | 238 | 3.7% | pre, force gather | gather | 1.2 | 1.287 | 0.117 | 1.066 |  | 1.0 | True |
| kw=vehicle | 238 | 3.7% | pre, force full+mask | full | 9.68 | 10.25 | 0.174 | 9.477 |  | 1.0 | True |
| kw=vehicle | 238 | 3.7% | post (exhaustive) | full | 38.194 | 78.872 |  | 9.429 | 28.497 | 1.0 | True |
| kw=vehicle | 238 | 3.7% | post-top100 | full+top100 | 17.314 | 18.057 |  | 9.414 | 7.697 | 0.23 | False |
| cpc=G06F & kw=signal | 14 | 0.2% | pre (auto path) | slices | 0.122 | 0.135 | 0.083 | 0.032 |  | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | pre, force gather | gather | 0.141 | 0.147 | 0.085 | 0.048 |  | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | pre, force full+mask | full | 9.61 | 10.459 | 0.184 | 9.405 |  | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | post (exhaustive) | full | 26.25 | 30.53 |  | 9.42 | 16.503 | 1.0 | True |
| cpc=G06F & kw=signal | 14 | 0.2% | post-top100 | full+top100 | 10.064 | 11.282 |  | 9.419 | 0.414 | 0.045 | False |

## Patent → similar (stage 1 doc vectors, stage 2 max-sim over 50 candidates), no hydration

| scenario | p50 ms | p95 ms | stage 1 p50 | stage 2 p50 |
|---|---|---|---|---|
| none | 1.155 | 3.307 | 0.256 | 0.869 |
| cpc=G06F | 0.968 | 1.37 | 0.026 | 0.883 |
| cpc=H01M | 0.913 | 1.328 | 0.017 | 0.877 |
| kw=vehicle | 0.932 | 1.422 | 0.027 | 0.813 |

## End-to-end per call (model loaded; includes query embedding and SQLite hydration of results + evidence)

| input | p50 ms | p95 ms | hydrate p50 |
|---|---|---|---|
| text | 7.73 | 7.99 | 0.31 |
| claim (by ref) | 10.58 | 11.17 | 0.4 |
| similar | 1.32 | 3.58 | 0.15 |
