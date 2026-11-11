# Part 1 benchmark — search latency with vs without hybrid filters

- machine: Apple M4 · Darwin arm64 · numpy 2.5.4
- index: build `acaaaff996cb`, 636 patents, 24,057 chunk vectors × 384 dims (float32, 37 MB), model `BAAI/bge-small-en-v1.5`
- BLAS threads: VECLIB_MAXIMUM_THREADS=1; reps per cell 200 (after 5 warm-up calls), k=10; times are wall-clock `perf_counter_ns`

## Query embedding (model inference, independent of filters)

p50 **6.09 ms**, p95 6.95 ms on `mps` (text queries; patent→similar and claim-by-reference reuse stored vectors and need no model call).

## Retrieval on the real index (filter + score + aggregate + top-k), text & claim query vectors

| scenario | docs kept | rows kept | strategy | path | p50 ms | p95 ms | filter p50 | score p50 | post-filter p50 | recall@10 vs pre | same top-10 as pre |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 636 | 100.0% | pre (auto path) | full | 0.626 | 1.522 | 0.0 | 0.602 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | pre (auto path) | slices | 0.316 | 0.441 | 0.078 | 0.223 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | pre, force gather | gather | 0.767 | 0.84 | 0.08 | 0.672 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | pre, force full+mask | full | 0.626 | 0.705 | 0.074 | 0.536 |  | 1.0 | True |
| cpc=B60C | 318 | 48.5% | post (exhaustive) | full | 0.568 | 0.634 |  | 0.518 | 0.011 | 1.0 | True |
| cpc=B60C | 318 | 48.5% | post-top100 | full+top100 | 0.618 | 0.769 |  | 0.542 | 0.049 | 0.965 | False |
| cpc=B60B | 294 | 47.8% | pre (auto path) | slices | 0.296 | 0.361 | 0.07 | 0.212 |  | 1.0 | True |
| cpc=B60B | 294 | 47.8% | pre, force gather | gather | 0.751 | 0.842 | 0.072 | 0.665 |  | 1.0 | True |
| cpc=B60B | 294 | 47.8% | pre, force full+mask | full | 0.619 | 0.681 | 0.067 | 0.537 |  | 1.0 | True |
| cpc=B60B | 294 | 47.8% | post (exhaustive) | full | 0.585 | 0.737 |  | 0.519 | 0.026 | 1.0 | True |
| cpc=B60B | 294 | 47.8% | post-top100 | full+top100 | 0.577 | 0.7 |  | 0.515 | 0.04 | 0.74 | False |
| cpc=B60D | 20 | 3.3% | pre (auto path) | slices | 0.033 | 0.034 | 0.012 | 0.013 |  | 1.0 | True |
| cpc=B60D | 20 | 3.3% | pre, force gather | gather | 0.057 | 0.06 | 0.012 | 0.037 |  | 1.0 | True |
| cpc=B60D | 20 | 3.3% | pre, force full+mask | full | 0.551 | 0.611 | 0.018 | 0.523 |  | 1.0 | True |
| cpc=B60D | 20 | 3.3% | post (exhaustive) | full | 0.763 | 0.85 |  | 0.519 | 0.219 | 1.0 | True |
| cpc=B60D | 20 | 3.3% | post-top100 | full+top100 | 0.603 | 0.805 |  | 0.534 | 0.041 | 0.135 | False |
| pub_date>=2025-02-01 | 130 | 23.6% | pre (auto path) | slices | 0.098 | 0.109 | 0.05 | 0.038 |  | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | pre, force gather | gather | 0.296 | 0.395 | 0.051 | 0.235 |  | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | pre, force full+mask | full | 0.635 | 0.735 | 0.069 | 0.543 |  | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | post (exhaustive) | full | 0.582 | 0.645 |  | 0.537 | 0.019 | 1.0 | True |
| pub_date>=2025-02-01 | 130 | 23.6% | post-top100 | full+top100 | 0.577 | 0.71 |  | 0.531 | 0.022 | 1.0 | True |
| kw=bicycle | 36 | 6.0% | pre (auto path) | slices | 0.077 | 0.087 | 0.02 | 0.049 |  | 1.0 | True |
| kw=bicycle | 36 | 6.0% | pre, force gather | gather | 0.096 | 0.109 | 0.022 | 0.064 |  | 1.0 | True |
| kw=bicycle | 36 | 6.0% | pre, force full+mask | full | 0.645 | 0.826 | 0.047 | 0.578 |  | 1.0 | True |
| kw=bicycle | 36 | 6.0% | post (exhaustive) | full | 22.258 | 27.313 |  | 0.532 | 21.699 | 1.0 | True |
| kw=bicycle | 36 | 6.0% | post-top100 | full+top100 | 8.255 | 8.742 |  | 0.518 | 7.693 | 0.26 | False |
| title=TIRE | 47 | 5.9% | pre (auto path) | slices | 0.083 | 0.094 | 0.014 | 0.062 |  | 1.0 | True |
| title=TIRE | 47 | 5.9% | pre, force gather | gather | 0.086 | 0.094 | 0.015 | 0.062 |  | 1.0 | True |
| title=TIRE | 47 | 5.9% | pre, force full+mask | full | 0.578 | 0.726 | 0.025 | 0.541 |  | 1.0 | True |
| title=TIRE | 47 | 5.9% | post (exhaustive) | full | 0.605 | 0.726 |  | 0.532 | 0.03 | 1.0 | True |
| title=TIRE | 47 | 5.9% | post-top100 | full+top100 | 0.597 | 0.735 |  | 0.547 | 0.024 | 0.605 | False |
| cpc=B60B & kw=bicycle | 34 | 5.8% | pre (auto path) | slices | 0.081 | 0.093 | 0.026 | 0.047 |  | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | pre, force gather | gather | 0.096 | 0.111 | 0.027 | 0.06 |  | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | pre, force full+mask | full | 0.573 | 0.648 | 0.037 | 0.523 |  | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | post (exhaustive) | full | 6.491 | 13.002 |  | 0.521 | 5.925 | 1.0 | True |
| cpc=B60B & kw=bicycle | 34 | 5.8% | post-top100 | full+top100 | 2.58 | 7.654 |  | 0.519 | 2.018 | 0.235 | False |

## Patent → similar (stage 1 doc vectors, stage 2 max-sim over 50 candidates), no hydration

| scenario | p50 ms | p95 ms | stage 1 p50 | stage 2 p50 |
|---|---|---|---|---|
| none | 0.739 | 0.795 | 0.041 | 0.698 |
| cpc=B60B | 0.791 | 0.824 | 0.027 | 0.698 |
| cpc=B60D | 0.29 | 0.313 | 0.01 | 0.269 |
| kw=bicycle | 0.529 | 0.564 | 0.011 | 0.495 |

## End-to-end per call (model loaded; includes query embedding and SQLite hydration of results + evidence)

| input | p50 ms | p95 ms | hydrate p50 |
|---|---|---|---|
| text | 7.3 | 7.86 | 0.29 |
| claim (by ref) | 0.9 | 1.25 | 0.29 |
| similar | 0.88 | 0.98 | 0.12 |

## Synthetic scale-up (random unit vectors; real per-patent chunk counts and CPC mix; docs contiguous by subclass)

| rows | scenario | rows kept | strategy | path | p50 ms | p95 ms | recall@10 vs pre |
|---|---|---|---|---|---|---|---|
| 1,000,000 | none | 100.0% | pre (auto path) | full | 22.14 | 22.62 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | pre (auto path) | slices | 11.5 | 11.91 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | pre, force gather | gather | 41.17 | 47.57 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | post (exhaustive) | full | 24.72 | 29.59 | 1.0 |
| 1,000,000 | cpc=B60C | 50.6% | post-top100 | full | 22.15 | 22.96 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | pre (auto path) | slices | 0.78 | 1.11 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | pre, force gather | gather | 2.46 | 2.85 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | post (exhaustive) | full | 23.23 | 24.01 | 1.0 |
| 1,000,000 | cpc=B60D | 2.9% | post-top100 | full | 22.17 | 23.35 | 0.33 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | pre (auto path) | gather | 4.6 | 5.57 | 1.0 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | pre, force gather | gather | 4.26 | 4.47 | 1.0 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | post (exhaustive) | full | 23.23 | 24.15 | 1.0 |
| 1,000,000 | scattered 5% (keyword-like) | 5.0% | post-top100 | full | 22.08 | 22.5 | 0.49 |
| 1,000,000 | scattered 0.5% | 0.5% | pre (auto path) | gather | 0.2 | 0.22 | 1.0 |
| 1,000,000 | scattered 0.5% | 0.5% | pre, force gather | gather | 0.2 | 0.2 | 1.0 |
| 1,000,000 | scattered 0.5% | 0.5% | post (exhaustive) | full | 23.48 | 23.68 | 1.0 |
| 1,000,000 | scattered 0.5% | 0.5% | post-top100 | full | 22.32 | 24.09 | 0.04 |
