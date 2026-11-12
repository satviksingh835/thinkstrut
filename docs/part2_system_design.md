# Part 2 — patsearch at 10 million patents: system design

**Audience:** an engineer (e.g. an intern) who will build this. Read Part 1's README first; this
document reuses its vocabulary (chunks, best-chunk ranking, pre-filtering, scoring paths).

**TL;DR.** Keep Part 1's search semantics exactly, swap its single-machine pieces for managed services:
raw and processed data in **S3**, metadata/text/pipeline state in **Postgres**, keyword + filter +
vector search in **OpenSearch** (one chunk-level vector index with filtering inside the ANN search,
1-bit compressed so 378 M vectors fit in RAM). A batch pipeline (parse → embed → load → index →
verify) driven by a job table in Postgres keeps everything idempotent and observable. Expected cost:
**~$50–200 one-off** to process and embed the backfill, **~$2.5–4 k/month** to run (≈$1 k/month for
a budget tier that trades away recall).

---

## 1. Scale: what "10 M patents" means

Per-patent figures are **measured** on the Part 1 build (636 patents, `artifacts/<build>/manifest.json`).

| Quantity | Per patent (measured) | At 10 M patents |
|---|---|---|
| Chunks (abstract + claims + description groups) | 37.8 | **378 M** |
| Tokens to embed | 5,933 | **59 B** |
| Raw JSON | 24.9 KB | 250 GB |
| Text + metadata + keyword index (SQLite, incl. indexes) | 37.7 KB | ~380 GB |
| Chunk vectors, 384-d, float32 / float16 / 1-bit | 58 KB / 29 KB / 1.8 KB | 581 GB / 290 GB / **18 GB** |
| CPU: load + clean + claims + chunk | 12.4 ms | 34 CPU-hours |
| Embedding on an M4 laptop GPU (measured 31.9 k tokens/s) | 0.19 s | 516 hours |

Assumptions: ~10 k new applications per week after the backfill (US publishes ~7 k/week); low query
load (examiners, ≤ 20 queries/s peak); target latency p50 < 150 ms, p95 < 500 ms.

**The number that drives the design: 378 M chunk vectors.** Brute force (Part 1's `ExactIndex`) would
read 145–581 GB per query; an uncompressed ANN index needs 340–630 GB of RAM.

## 2. Key decision: chunk-level ANN, not per-patent summary vectors

The cheap alternative is two-stage: an ANN index over one (or a few) summary vectors per patent
(10–30 M vectors, fits one Postgres+pgvector box), then exact re-scoring of the top-N candidates'
chunks. I measured how much it loses against Part 1's exact chunk search (20 benchmark queries,
recall@10 = share of the exact top-10 recovered):

| Stage-1 vectors per patent | N=10 | N=20 | N=50 | N=100 |
|---|---|---|---|---|
| 1 — mean(abstract + claims) | 0.585 | 0.790 | 0.880 | 0.920 |
| 1 — mean(all chunks) | 0.645 | 0.815 | 0.900 | 0.955 |
| 3 — abstract / claims / description (max) | 0.620 | 0.805 | 0.910 | 0.965 |

Even with N = 100 candidates — **16% of this 636-patent corpus** — it misses 4–8% of the true top-10;
at 10 M patents, N = 1,000 is 0.01% of the corpus, so it would miss far more. Summary vectors blur
a patent into one point, while a query usually matches one specific claim or paragraph. **Decision:**
index every chunk (as Part 1 does) and pay for memory with compression. The two-stage design is kept
as the documented *budget tier* (§7).

## 3. Architecture

```text
                    ┌──────────────── ingestion (batch, weekly + one-off backfill) ────────────────┐
 USPTO / provider   │                                                                              │
 weekly files ──► register ──► parse/clean/chunk ──► embed ──► load DB ──► index ──► verify      │
                    │  (CPU)        (CPU, Part 1 code)   (GPU spot     (CPU)      (CPU)     (canaries)│
                    │                                     or CPU)                                  │
                    └──────┬──────────────┬──────────────────┬───────────┬──────────┬──────────────┘
                           ▼              ▼                  ▼           ▼          ▼
                    S3 raw/        S3 processed/      S3 embeddings/  Postgres   OpenSearch
                    (immutable)    (parquet)          (fp16 parquet)  (records,  (patents index:
                                                                      text, jobs  BM25 + filters;
                                                                      + status)   chunks index:
                                                                                  vectors + filters)
 query path:  client ─► API (FastAPI, CPU) ─► embed query (ONNX) ─► OpenSearch ─► Postgres (text) ─► JSON
```

| Component | Technology | Role | Why this choice |
|---|---|---|---|
| Object store | S3 | Raw files as received; processed parquet; embeddings per model version | Cheap, durable; lets you rebuild every downstream store without re-parsing or re-embedding |
| System of record + job queue | Postgres (RDS) | Patents, claims, chunk text, aliases; `batches` / `jobs` / `index_versions` tables | Transactions for idempotent upserts; `SELECT … FOR UPDATE SKIP LOCKED` makes it a job queue — one less service |
| Search | OpenSearch (managed) with k-NN plugin | `patents` index: title/abstract text, CPC, dates, title keys. `chunks` index: one doc per chunk = vector + copied filter fields | Keyword (BM25), structured filters and ANN in one engine; filters are applied *inside* the ANN search ("efficient filtering"), falling back to exact search when the filtered set is small — the same cost-based choice as Part 1's `choose_path` |
| Workers | Containers on ECS/AWS Batch; GPU spot for backfill embedding | Run pipeline stages | Stateless; scale to zero between weekly runs |
| API | Part 1 FastAPI app, on CPU containers | Query embedding (ONNX bge-small), query planning, grouping, hydration | Same response JSON as Part 1, so the UI and CLI work unchanged |
| Monitoring | CloudWatch / Grafana | Pipeline + query metrics, alerts | §9 |

**Part 1 code reuse.**

| Part 1 module | At scale |
|---|---|
| `load.py`, `normalize.py`, `claims.py`, `chunk.py`, `tokens.py` | Run unchanged inside the *parse* worker, one batch at a time (the loader already streams files) |
| `embed.py` | The *embed* worker (GPU) and the API (export to ONNX for CPU) |
| `filters.py` (`FilterSpec`, `compile_filter`) | Unchanged API contract; add a translator to the OpenSearch query DSL next to `filter_sql` |
| `vindex.py` (`VectorIndex` protocol) | New `OpenSearchIndex` implementing `top_docs` / `top_rows` / `doc_row_scores` |
| `search.py` (`SearchEngine`, `SearchResponse`) | Same orchestration and response type; storage calls go to OpenSearch/Postgres |
| `store.py` (SQLite) | Replaced by Postgres + S3 |

## 4. Data model

**S3 layout** (every key is deterministic, so re-running a stage overwrites, never duplicates):
```text
s3://patsearch/raw/<pub_week>/<source_file>                     as received, never modified
s3://patsearch/processed/<batch_id>/patents.parquet             cleaned patents (Part 1 Patent fields + flags)
s3://patsearch/processed/<batch_id>/chunks.parquet              chunk_id, doc_number, kind, label, embed_text, chunk_hash
s3://patsearch/processed/<batch_id>/rejects.jsonl               dropped records + reason
s3://patsearch/embeddings/<model_version>/<batch_id>.parquet    chunk_id, chunk_hash, vector (float16)
```
`chunk_id = "<doc_number>:<ordinal>"`; `chunk_hash = sha1(embed_text)` (embedding-cache key: a
chunk whose text did not change is never re-embedded).

**Postgres** (abbreviated DDL):
```sql
CREATE TABLE batches (
  batch_id      bigserial PRIMARY KEY,
  source_uri    text NOT NULL,
  sha256        text NOT NULL UNIQUE,       -- registering the same file twice is a no-op
  pub_week      date,
  state         text NOT NULL DEFAULT 'registered',
                -- registered | parsed | embedded | loaded | indexed | verified | needs_review | failed
  stats         jsonb,                      -- Part 1 LoadStats, claim/chunk counts, per-stage timings
  model_version text,
  created_at    timestamptz DEFAULT now(),
  updated_at    timestamptz DEFAULT now()
);
CREATE TABLE jobs (
  job_id       bigserial PRIMARY KEY,
  batch_id     bigint NOT NULL REFERENCES batches,
  stage        text NOT NULL,               -- parse | embed | load | index | verify
  status       text NOT NULL DEFAULT 'queued',  -- queued | running | done | failed
  attempts     int  NOT NULL DEFAULT 0,
  not_before   timestamptz DEFAULT now(),   -- retry backoff
  worker       text, heartbeat_at timestamptz, last_error text,
  UNIQUE (batch_id, stage)
);
CREATE TABLE patents (
  doc_number    text PRIMARY KEY,
  canonical     text REFERENCES patents,    -- set for aliases (A9 republication -> its A1)
  kind text, title text, abstract text, pub_date date,
  cpc_codes     text[],                     -- real data has several codes per patent
  has_description boolean, flags jsonb,
  content_hash  text,                       -- Part 1 dedupe key
  batch_id      bigint REFERENCES batches,
  version       int NOT NULL DEFAULT 1,     -- bumped when a corrected publication replaces it
  updated_at    timestamptz DEFAULT now()
);
CREATE INDEX ON patents (content_hash);
CREATE TABLE claims (doc_number text, lo int, hi int, label text, exact bool, depends_on int, text text,
                     PRIMARY KEY (doc_number, lo));
CREATE TABLE chunks (chunk_id text PRIMARY KEY, doc_number text, kind smallint, label text, text text);
CREATE INDEX ON chunks (doc_number);
CREATE TABLE index_versions (index_name text PRIMARY KEY, model_version text, n_patents bigint,
                             n_chunks bigint, eval jsonb, live boolean, created_at timestamptz);
```
A worker claims a job with:
```sql
UPDATE jobs SET status = 'running', attempts = attempts + 1, worker = $1, heartbeat_at = now()
WHERE job_id = (SELECT job_id FROM jobs
                WHERE status = 'queued' AND stage = $2 AND not_before <= now()
                ORDER BY job_id FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING *;
```

**OpenSearch** (sketch; check field options against the deployed version — on-disk/compressed
vectors need OpenSearch ≥ 2.17):
```jsonc
// patents index — 10 M docs, ~4 primary shards
{ "doc_number": "keyword", "title": "text", "abstract": "text",  // analyzer: standard tokenizer + lowercase
  "title_norm": "keyword", "cpc_codes": "keyword", "cpc_groups": "keyword", // + asciifolding, NO stemming
  "cpc_subclasses": "keyword", "pub_date": "date", "batch_id": "integer" }  // (= Part 1 FTS5 semantics)

// chunks index — 378 M docs, ~18–24 primary shards (20–40 GB each), _source excludes the vector
{ "chunk_id": "keyword", "doc_number": "keyword", "kind": "byte", "batch_id": "integer",
  "cpc_codes": "keyword", "cpc_groups": "keyword", "cpc_subclasses": "keyword", "pub_date": "date",
  "vector": { "type": "knn_vector", "dimension": 384, "space_type": "innerproduct",
              "mode": "on_disk", "compression_level": "32x" } }   // 1-bit in RAM, full precision on disk for rescoring
```
Small filter fields (CPC, date) are copied onto every chunk so they can be applied inside the ANN
search. Title/abstract text is **not** copied (that would mean indexing it 38 times); keyword
filters are resolved on the `patents` index — see §6.

## 5. Ingestion pipeline

One **batch** = one weekly source file (backfill: shards of ~10 k patents → ~1,000 batches). Each
stage reads the previous stage's S3 output, writes its own to a deterministic location, and enqueues
the next stage's job. A stage can be re-run at any time without side effects.

| # | Stage | Does | Output | Compute | Failure handling |
|---|---|---|---|---|---|
| 1 | register | New file → copy to `raw/`, sha256, insert `batches` row | `batches` row | trivial | Duplicate sha256 → skip (already ingested) |
| 2 | parse | Part 1 `load` → `dedupe` → `segment_claims` → `chunk_patent` | `processed/<batch>/…parquet`, `rejects.jsonl`, stats | ~12 ms/patent/core | Bad records → `rejects.jsonl`, batch continues; drop rate > 1% → `needs_review` |
| 3 | embed | Embed chunks missing from the cache for `(model_version, chunk_hash)` | `embeddings/<model>/<batch>.parquet` | GPU (backfill), CPU (weekly) | CUDA OOM → halve batch size and retry; check count = chunks, no NaN, ‖v‖ ≈ 1 |
| 4 | load | One transaction per batch: upsert patents / claims / chunks; cross-batch dedupe on `content_hash` (A9 → alias); corrections bump `version` | Postgres rows | CPU, minutes | Transaction rolls back as a whole; retry |
| 5 | index | Bulk-index into `patents_vN` and `chunks_vN` with `_id` = doc_number / chunk_id | OpenSearch docs | CPU | Bulk API reports per-item errors → retry only failed items; delete-by-query old chunks of re-published patents |
| 6 | verify | Count check (`_count` by batch) + canaries: for 20 random patents of the batch, claim 1 → own patent at rank 1 (Part 1 test, run live) | batch `verified` | seconds | Failure → `needs_review`, batch stays invisible (see below) |

**Visibility.** Every patent carries `batch_id`; queries add `batch_id ∈ verified` implicitly (a cached
set), so half-indexed batches are never served. Responses report `index_as_of` = newest verified week.

**Backfill vs weekly.** Backfill: create the indexes with `number_of_replicas: 0` and
`refresh_interval: -1`, run ~10 GPU spot workers (≈ 11 h wall-clock, §7), then enable 1 replica and
refresh. Weekly: the same pipeline on ~1 file, embedding on CPU is fine (~60 M tokens).

**Model change / re-index (blue-green).** New `model_version` → re-embed everything into a new
`embeddings/<model>/` prefix → build `patents_vN+1`, `chunks_vN+1` → run the evaluation set →
switch the `patents` / `chunks` aliases atomically → keep the old indexes a week for rollback. Never
mix vectors of two models in one index (Part 1's `SearchEngine.embedder` check enforces this in code).

## 6. Query pipeline

Same three input types and the same response JSON as Part 1.

1. **Parse filters** with Part 1's `compile_filter` (400 on bad input).
2. **Query vector.** Text: ONNX bge-small on CPU (~10 ms), LRU cache by query string.
   Patent / claim: fetch stored vectors (no model call).
3. **Resolve the filters.** CPC and date go straight into the ANN filter (copied fields).
   Keyword / exact-title / title-phrase run first on the small `patents` index:
   - 0 matches → empty result.
   - ≤ 65 k patents (OpenSearch's `terms` limit) → push their doc_numbers into the ANN filter.
   - more → no push-down; oversample the ANN request
     (`k' = min(10 000, 3·k / selectivity)`) and keep chunks whose patent passes — the selective cases
     use the exact path, the broad ones lose little because many candidates pass. This is Part 1's
     pre- vs post-filter trade-off, decided per query from the measured selectivity.
4. **ANN over `chunks`** (`k_chunks = 20 × k`, filtered). OpenSearch uses exact search when the
   filtered set is small and filtered HNSW otherwise, rescoring the 1-bit candidates with full
   vectors.
5. **Group by patent**: the first hit of each doc_number is its best chunk (Part 1 ranking); if fewer
   than k patents appear, repeat with a larger `k_chunks`.
6. **Evidence**: one more ANN call restricted to the top-k doc_numbers (~380 chunks, exact) → best
   claim / paragraph / abstract per patent.
7. **Hydrate** titles and evidence text from Postgres by primary key.

*Similar patents*: one ANN request per source vector (abstract + independent claims, ≤ 8, sent as a
multi-search), union the candidate patents, score each by the mean of its best match per source
vector (Part 1's max-sim). *Claim mapping*: the stored claim vector goes through steps 3–7.

| Step | Expected latency |
|---|---|
| Query embedding (text only, CPU) | 5–15 ms |
| Keyword resolution on `patents` (when present) | 5–30 ms |
| Filtered ANN + rescoring on `chunks` | 20–100 ms |
| Evidence ANN (top-k only) | 5–20 ms |
| Postgres hydration | 5–10 ms |
| **Total** | **~40–180 ms** (p95 target 500 ms) |

## 7. Cost

Approximate AWS on-demand list prices (us-east-1, ±30%; check the AWS pricing calculator; reserved
instances cut 30–40%).

**One-off backfill (10 M patents)**

| Item | Estimate | Basis |
|---|---|---|
| Parse / clean / chunk | ~$5 | 34 CPU-hours on spot |
| Embedding, self-hosted bge-small | **~$30–200** | 59 B tokens; ~150 k tokens/s per T4 (≈5× the measured M4) → ~110 GPU-hours (≈11 h on 10 spot GPUs at ~$0.2–0.5/h); pessimistic 3× slower = ~$200 |
| Embedding via a hosted API instead | ~$1,200 | at ~$0.02 per 1 M tokens — and a different model than Part 1 |
| OpenSearch / Postgres bulk load | in the monthly price | indexing takes hours to a day on the cluster |

**Monthly, steady state**

| Item | Recommended tier | Notes |
|---|---|---|
| OpenSearch data nodes | $1,500–2,900 | ~70 GB of k-NN memory (378 M × (48 B 1-bit vector + ~128 B HNSW links)) → 3 × 64 GB nodes; 128 GB nodes for headroom |
| OpenSearch masters + storage | ~$450 | 3 small dedicated masters; ~1.5 TB gp3 (full-precision vectors on disk for rescoring, 1 replica) |
| Postgres (RDS, 16 GB, ~500 GB gp3, backups) | ~$270 | text/metadata/state only — no vectors here |
| API (2 small CPU containers) | ~$110 | |
| S3 (~650 GB: raw + parquet + fp16 embeddings) | ~$15 | |
| Weekly pipeline workers | ~$10 | ~1 h of CPU per week |
| Monitoring, logs, misc | ~$100 | |
| **Total** | **~$2.5–4 k/month** | |

**Budget tier (~$1 k/month):** Postgres + pgvector only — one HNSW index over 10–30 M summary
vectors + exact re-scoring of the top-N candidates' chunks (chunk vectors stored as packed bytes per
patent). Costs a third, but loses recall (§2); acceptable for a demo, not for prior-art search.

**Cost levers:** fewer vectors (merge description paragraphs into ~500-token chunks: ~-30% vectors;
re-measure recall); reserved instances; IVF-PQ instead of HNSW; drop the replica (no HA).

## 8. Error handling

| Where | Failure | Handling |
|---|---|---|
| Records | missing / null / wrong-type fields, unparseable claims | Part 1 tolerant loader: empty value + flag; drop only without ID or text; rejects to S3 with reason |
| Batch quality | drop rate > 1%, sudden shift in flag mix (e.g. % of fully-exact claims halves) | batch → `needs_review`, not served, alert (catches upstream parser changes like the claim fragmentation) |
| Stage execution | transient errors (network, S3, DB timeout, spot interruption) | job retried with exponential backoff (`not_before`), max 3 attempts, then `failed` + alert; a reaper re-queues `running` jobs whose heartbeat is > 10 min old |
| Embedding | OOM, NaN vectors | halve the batch and retry; NaN / ‖v‖ ≠ 1 fails the stage |
| Indexing | partial bulk failures, mapping errors | retry failed items only; mapping errors fail the stage (code bug, not data) |
| Cross-store consistency | Postgres and OpenSearch drift (no distributed transactions) | idempotent upserts keyed by doc_number / chunk_id; nightly reconciliation compares per-batch counts and `content_hash` between Postgres and both indexes, re-queues `index` for mismatches |
| Query: bad input | invalid CPC / date / scope | 400 with Part 1's message |
| Query: dependencies | OpenSearch slow / down; embedding model down | timeout + one retry with smaller `k`; 503 with a clear message; patent- and claim-based queries still work without the model (stored vectors) |
| Query: partial data | a batch not yet verified | not visible; `index_as_of` in every response |

## 9. Tracking contents and status

- **Pipeline status**: the `batches` and `jobs` tables *are* the status. Dashboard: batches per
  state, oldest non-verified batch, failed jobs with `last_error`, throughput per stage (patents/s,
  tokens/s), backfill ETA.
- **Contents**: `patents` per CPC subclass and per publication week (gaps = a missing weekly file);
  data-quality per batch from `stats` (drop rate, % without description, % fully-exact claims,
  truncated chunks — all already computed by Part 1's build); `index_versions` (which model and index
  are live, their counts and evaluation scores).
- **Search health**: query p50/p95 per input type, QPS, error rate, result-count-zero rate,
  filter selectivity distribution, cache hit rate.
- **Relevance canaries**: a fixed evaluation set run nightly (known-item tests: a patent's own claim
  must return it at rank 1; Part 1's benchmark queries with expected top results), recall@10 tracked
  per index version.
- **Alerts**: job `failed`; batch `needs_review`; ingestion lag > 8 days; Postgres/OpenSearch count
  mismatch; query p95 > 500 ms for 10 min; canary recall drop > 5 points.

## 10. Major challenges at scale

1. **Vector memory.** 378 M vectors only fit in RAM with 1-bit compression; that costs recall, which
   rescoring with full vectors from disk must recover — tune oversampling and `ef_search` against the
   canary set. Vector count is the main cost driver.
2. **Filtered ANN correctness.** Filtering after an ANN search under-fills results (Part 1: recall
   0.06–0.38 for a post-filter on the top-100 with selective filters). Filters must run inside the
   ANN search; broad keyword filters (that can't be pushed down) remain the weak spot.
3. **Keyword filters span two indexes.** Title/abstract live on `patents`, vectors on `chunks`; the
   doc-id push-down is capped (~65 k), so very broad keyword + vector queries rely on oversampling.
   Fix later with frequent-filter bitset caching or by indexing a compact keyword field on chunks.
4. **Rebuild cost and time.** A model change means re-embedding 59 B tokens and rebuilding the index
   (hours to a day, double storage during the switch). The embedding cache and blue-green aliases
   make it routine, not risky.
5. **Messier real data.** Several CPC codes per patent, other sources and XML formats, non-English
   text (Part 3 translation track), corrections and continuations. The tolerant loader and per-batch
   quality checks are the first line of defence; dedupe on content hash only covers exact
   republications.
6. **No relevance labels.** Without an evaluation set, quantization and ANN tuning are blind; build
   one early (examiner citations as positives, known-item tests).
7. **Consistency across S3, Postgres and OpenSearch** without distributed transactions — handled with
   deterministic IDs, idempotent stages, the verified-batch gate and nightly reconciliation.

## 11. How to start (intern plan)

1. **Local stack** (`docker-compose`): Postgres, single-node OpenSearch with the k-NN plugin, MinIO
   (S3 API). Create the schema above.
2. **Pipeline skeleton**: the `jobs` table + a worker loop (claim job → run stage → enqueue next).
   Port Part 1's build steps into `parse`, `embed`, `load`, `index` stage functions that read and write
   the S3 layout above.
3. **Query API**: an `OpenSearchIndex` implementing Part 1's `VectorIndex` protocol, plus the filter
   → OpenSearch DSL translator. Keep the Part 1 response JSON so the web UI works unchanged.
4. **Equivalence check**: run both engines on the 640-patent sample; top-10 overlap with Part 1
   should be ≥ 0.95 (differences come only from ANN + quantization). Port Part 1's tests.
5. **Scale test** on 100 k–1 M patents (the Part 3 large dataset): measure throughput per stage,
   index memory, latency; tune shards, oversampling, `ef_search`; then size the production cluster
   from measured numbers instead of the estimates in §7.
6. **Backfill**, then switch on the weekly schedule, dashboards and alerts.
