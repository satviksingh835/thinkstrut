# Part 1 benchmark — search latency with vs without hybrid filters

- machine: Apple M4 · Darwin arm64 · numpy 2.5.4
- index: build `acaaaff996cb`, 636 patents, 24,057 chunk vectors × 384 dims (float32, 37 MB), model `BAAI/bge-small-en-v1.5`
- BLAS threads: VECLIB_MAXIMUM_THREADS=default; reps per cell 200 (after 5 warm-up calls), k=10; times are wall-clock `perf_counter_ns`

## Query embedding (model inference, independent of filters)

p50 **5.98 ms**, p95 6.29 ms on `mps` (text queries; patent→similar and claim-by-reference reuse stored vectors and need no model call).

## Retrieval on the real index (filter + score + aggregate + top-k), text & claim query vectors

| scenario | docs kept | rows kept | strategy | path | p50 ms | p95 ms | filter p50 | score p50 | post-filter p50 | recall@10 vs pre | same top-10 as pre |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 636 | 100.0% | pre (auto path) | full | 0.527 | 0.544 | 0.0 | 0.509 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | pre (auto path) | slices | 0.288 | 0.337 | 0.069 | 0.206 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | pre, force gather | gather | 0.754 | 0.806 | 0.078 | 0.663 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | pre, force full+mask | full | 0.622 | 0.659 | 0.073 | 0.535 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | post (exhaustive) | full | 0.597 | 0.689 |  | 0.537 | 0.013 | 1.0 | True |
| cpc=B60C | 318 | 48.5% | post-top100 | full+top100 | 0.596 | 0.667 |  | 0.529 | 0.044 | 0.965 | False |
| cpc=B60B | 294 | 47.8% | pre (auto path) | slices | 0.271 | 0.301 | 0.064 | 0.193 |  | 1.0 | True |
| cpc=B60B | 294 | 47.8% | pre, force gather | gather | 0.724 | 0.763 | 0.07 | 0.641 |  | 1.0 | True |
| cpc=B60B | 294 | 47.8% | pre, force full+mask | full | 0.623 | 0.652 | 0.068 | 0.54 |  | 1.0 | True |
| cpc=B60B | 294 | 47.8% | post (exhaustive) | full | 0.571 | 0.646 |  | 0.517 | 0.026 | 1.0 | True |
| cpc=B60B | 294 | 47.8% | post-top100 | full+top100 | 0.576 | 0.61 |  | 0.516 | 0.039 | 0.74 | False |
| cpc=B60D | 20 | 3.3% | pre (auto path) | slices | 0.033 | 0.037 | 0.012 | 0.013 |  | 1.0 | True |
| cpc=B60D | 20 | 3.3% | pre, force gather | gather | 0.057 | 0.06 | 0.012 | 0.037 |  | 1.0 | True |
| cpc=B60D | 20 | 3.3% | pre, force full+mask | full | 0.551 | 0.592 | 0.018 | 0.524 |  | 1.0 | True |
| cpc=B60D | 20 | 3.3% | post (exhaustive) | full | 0.756 | 0.798 |  | 0.516 | 0.213 | 1.0 | True |
| cpc=B60D | 20 | 3.3% | post-top100 | full+top100 | 0.573 | 0.601 |  | 0.518 | 0.034 | 0.135 | False |
| pub_date>=2025-02-01 | 130 | 23.6% | pre (auto path) | slices | 0.093 | 0.103 | 0.049 | 0.036 |  | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | pre, force gather | gather | 0.283 | 0.302 | 0.049 | 0.226 |  | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | pre, force full+mask | full | 0.608 | 0.637 | 0.065 | 0.531 |  | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | post (exhaustive) | full | 0.563 | 0.732 |  | 0.524 | 0.017 | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | post-top100 | full+top100 | 0.557 | 0.595 |  | 0.517 | 0.019 | 1.0 | True |
| kw=bicycle | 36 | 6.0% | pre (auto path) | slices | 0.074 | 0.079 | 0.02 | 0.048 |  | 1.0 | True |
| kw=bicycle | 36 | 6.0% | pre, force gather | gather | 0.091 | 0.096 | 0.021 | 0.062 |  | 1.0 | True |
| kw=bicycle | 36 | 6.0% | pre, force full+mask | full | 0.562 | 0.592 | 0.031 | 0.521 |  | 1.0 | True |
| kw=bicycle | 36 | 6.0% | post (exhaustive) | full | 21.932 | 26.22 |  | 0.517 | 21.384 | 1.0 | True |
| kw=bicycle | 36 | 6.0% | post-top100 | full+top100 | 8.13 | 8.533 |  | 0.522 | 7.578 | 0.26 | False |
| title=TIRE | 47 | 5.9% | pre (auto path) | slices | 0.083 | 0.085 | 0.014 | 0.062 |  | 1.0 | True |
| title=TIRE | 47 | 5.9% | pre, force gather | gather | 0.085 | 0.099 | 0.015 | 0.062 |  | 1.0 | True |
| title=TIRE | 47 | 5.9% | pre, force full+mask | full | 0.558 | 0.603 | 0.022 | 0.525 |  | 1.0 | True |
| title=TIRE | 47 | 5.9% | post (exhaustive) | full | 0.573 | 0.633 |  | 0.514 | 0.026 | 1.0 | True |
| title=TIRE | 47 | 5.9% | post-top100 | full+top100 | 0.554 | 0.58 |  | 0.515 | 0.019 | 0.605 | False |
| cpc=B60B & kw=bicycle | 34 | 5.8% | pre (auto path) | slices | 0.081 | 0.083 | 0.026 | 0.046 |  | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | pre, force gather | gather | 0.097 | 0.104 | 0.028 | 0.061 |  | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | pre, force full+mask | full | 0.569 | 0.594 | 0.035 | 0.524 |  | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | post (exhaustive) | full | 6.398 | 12.809 |  | 0.522 | 5.835 | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | post-top100 | full+top100 | 2.483 | 7.252 |  | 0.522 | 1.924 | 0.235 | False |

## Patent → similar (stage 1 doc vectors, stage 2 max-sim over 50 candidates), no hydration

| scenario | p50 ms | p95 ms | stage 1 p50 | stage 2 p50 |
|---|---|---|---|---|
| none | 0.724 | 0.76 | 0.04 | 0.682 |
| cpc=B60B | 0.785 | 0.809 | 0.026 | 0.692 |
| cpc=B60D | 0.287 | 0.299 | 0.009 | 0.266 |
| kw=bicycle | 0.52 | 0.547 | 0.011 | 0.488 |

## End-to-end per call (model loaded; includes query embedding and SQLite hydration of results + evidence)

| input | p50 ms | p95 ms | hydrate p50 |
|---|---|---|---|
| text | 7.03 | 7.36 | 0.26 |
| claim (by ref) | 0.85 | 0.92 | 0.27 |
| similar | 0.86 | 0.93 | 0.12 |

## Synthetic scale-up (random unit vectors; real per-patent chunk counts and CPC mix; docs contiguous by subclass)

| rows | scenario | rows kept | strategy | path | p50 ms | p95 ms | recall@10 vs pre |
|---|---|---|---|---|---|---|---|
| 100,000 | none | 100.0% | pre (auto path) | full | 2.18 | 2.21 | 1.0 |
| 100,000 | cpc=B60C | 48.2% | pre (auto path) | slices | 1.08 | 1.11 | 1.0 |
| 100,000 | cpc=B60C | 48.2% | pre, force gather | gather | 3.54 | 3.59 | 1.0 |
| 100,000 | cpc=B60C | 48.2% | post (exhaustive) | full | 2.26 | 2.43 | 1.0 |
| 100,000 | cpc=B60C | 48.2% | post-top100 | full | 2.19 | 2.22 | 1.0 |
| 100,000 | cpc=B60D | 2.8% | pre (auto path) | slices | 0.03 | 0.04 | 1.0 |
| 100,000 | cpc=B60D | 2.8% | pre, force gather | gather | 0.13 | 0.13 | 1.0 |
| 100,000 | cpc=B60D | 2.8% | post (exhaustive) | full | 2.31 | 2.34 | 1.0 |
| 100,000 | cpc=B60D | 2.8% | post-top100 | full | 2.21 | 2.4 | 0.33 |
| 100,000 | scattered 5% (keyword-like) | 5.2% | pre (auto path) | gather | 0.23 | 0.24 | 1.0 |
| 100,000 | scattered 5% (keyword-like) | 5.2% | pre, force gather | gather | 0.22 | 0.23 | 1.0 |
| 100,000 | scattered 5% (keyword-like) | 5.2% | post (exhaustive) | full | 2.3 | 2.34 | 1.0 |
| 100,000 | scattered 5% (keyword-like) | 5.2% | post-top100 | full | 2.19 | 2.23 | 0.56 |
| 100,000 | scattered 0.5% | 0.4% | pre (auto path) | slices | 0.04 | 0.05 | 1.0 |
| 100,000 | scattered 0.5% | 0.4% | pre, force gather | gather | 0.04 | 0.04 | 1.0 |
| 100,000 | scattered 0.5% | 0.4% | post (exhaustive) | full | 2.52 | 2.57 | 1.0 |
| 100,000 | scattered 0.5% | 0.4% | post-top100 | full | 2.19 | 2.22 | 0.03 |
| 1,000,000 | none | 100.0% | pre (auto path) | full | 21.89 | 22.38 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | pre (auto path) | slices | 11.21 | 11.83 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | pre, force gather | gather | 37.58 | 38.88 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | post (exhaustive) | full | 23.23 | 24.2 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | post-top100 | full | 21.97 | 22.51 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | pre (auto path) | slices | 0.66 | 0.69 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | pre, force gather | gather | 2.21 | 2.53 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | post (exhaustive) | full | 23.14 | 23.76 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | post-top100 | full | 21.96 | 23.36 | 0.38 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | pre (auto path) | gather | 4.54 | 7.3 | 1.0 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | pre, force gather | gather | 4.2 | 4.28 | 1.0 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | post (exhaustive) | full | 25.59 | 28.21 | 1.0 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | post-top100 | full | 22.51 | 24.93 | 0.57 |
| 1,000,000 | scattered 0.5% | 0.5% | pre (auto path) | gather | 0.23 | 0.26 | 1.0 |
| 1,000,000 | scattered 0.5% | 0.5% | pre, force gather | gather | 0.22 | 0.23 | 1.0 |
| 1,000,000 | scattered 0.5% | 0.5% | post (exhaustive) | full | 23.48 | 26.3 | 1.0 |
| 1,000,000 | scattered 0.5% | 0.5% | post-top100 | full | 21.92 | 22.47 | 0.06 |
