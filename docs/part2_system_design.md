# Part 2: System Design for Patent Search at 10 Million Documents

## 1. Summary

This document describes how the patent search engine built in Part 1 (the prototype) can operate on
a corpus of 10 million patents. The search behaviour is unchanged: each patent is split into chunks,
every chunk is embedded, a patent is ranked by its best-matching chunk, and hard filters are applied
before vector scoring. The single-machine components of the prototype are replaced by managed AWS
services:

- **Amazon S3** stores raw input files, processed data and embeddings.
- **Amazon RDS for PostgreSQL** stores patent records, text and pipeline state, and serves as the
  job queue.
- **Amazon OpenSearch Service** provides keyword search, structured filters and approximate
  nearest-neighbour (ANN) search over all 378 million chunk vectors, which are compressed to one bit
  per dimension in memory.

A batch pipeline (register, parse, embed, load, index, verify) processes one weekly file at a time.
Every stage is idempotent, and its status is recorded in PostgreSQL. The estimated cost is
USD 50–200 for the one-time backfill and USD 2,500–4,000 per month in steady state. A reduced
configuration at approximately USD 1,000 per month is possible at the cost of lower recall
(Section 9.3).

## 2. Scope and Assumptions

- **Corpus:** 10 million patent applications, with approximately 10,000 new applications per week
  after the initial backfill. The USPTO publishes roughly 7,000 applications per week.
- **Query load:** low concurrency (patent examiners), with at most 20 queries per second at peak.
- **Latency targets:** 150 ms at the median and 500 ms at the 95th percentile.
- **Functional scope:** the three query types of the prototype (text query, similar patents and
  claim mapping), with the same hybrid filters and the same response format.

## 3. Capacity Estimates

The per-patent figures below were measured on the prototype index (636 patents; see
`artifacts/<build>/manifest.json`) and scaled linearly.

| Quantity | Per patent (measured) | At 10 million patents |
|---|---|---|
| Chunks (abstract, claims, description paragraph groups) | 37.8 | 378 million |
| Tokens to embed | 5,933 | 59 billion |
| Raw JSON | 24.9 KB | 250 GB |
| Text, metadata and keyword index (SQLite, including indexes) | 37.7 KB | 380 GB |
| Chunk vectors, 384 dimensions: float32 / float16 / 1-bit | 58 KB / 29 KB / 1.8 KB | 581 GB / 290 GB / 18 GB |
| CPU time for loading, cleaning, claim re-assembly and chunking | 12.4 ms | 34 CPU-hours |
| Embedding time on an Apple M4 GPU (measured at 31,900 tokens/s) | 0.19 s | 516 hours |

The volume of chunk vectors determines the design. Exhaustive search, as performed by the
prototype, would read 145–581 GB per query, and an uncompressed ANN index would require 340–630 GB
of memory.

## 4. Design Decision: Index Granularity

Two options were considered for the vector index:

- **Option A (summary vectors):** index one or a few vectors per patent (10–30 million vectors in
  total), retrieve the top N candidate patents, and then re-score all chunks of those candidates
  exactly. A single PostgreSQL instance with pgvector can hold this index.
- **Option B (chunk vectors):** index every chunk, as the prototype does, and compress the vectors
  so that the index fits in memory.

Option A was evaluated against the exact chunk-level search of the prototype on the 20 benchmark
queries. The table reports recall@10, defined as the share of the exact top 10 results that the
two-stage method also returns.

| Summary vectors per patent | N = 10 | N = 20 | N = 50 | N = 100 |
|---|---|---|---|---|
| 1: mean of abstract and claim chunks | 0.585 | 0.790 | 0.880 | 0.920 |
| 1: mean of all chunks | 0.645 | 0.815 | 0.900 | 0.955 |
| 3: abstract, claims and description (maximum score) | 0.620 | 0.805 | 0.910 | 0.965 |

With 100 candidates, which is 16% of the 636-patent test corpus, Option A still misses 4–8% of the
correct results. At 10 million patents, a candidate list of 1,000 covers only 0.01% of the corpus,
so the loss would be considerably larger. A summary vector averages the entire patent, whereas a
query typically matches a single claim or paragraph. Option B is therefore adopted. Option A is
retained as a lower-cost configuration (Section 9.3).

## 5. Architecture

### 5.1 Overview

![Architecture: inside the AWS Cloud boundary, six AWS Batch ingestion jobs write to Amazon S3, Amazon RDS for PostgreSQL and Amazon OpenSearch Service; the Search API on ECS Fargate serves patent examiners from OpenSearch and RDS; EventBridge triggers ingestion and CloudWatch collects metrics](images/architecture.png)

**Ingestion (dark arrows).** Amazon EventBridge starts ingestion once a week, and the initial
backfill runs the same stages over approximately 1,000 batches. Each stage runs as an AWS Batch job,
claims its work from the job table in PostgreSQL and enqueues the next stage on completion.

1. **Register:** copies the new USPTO file to `raw/` in S3 and creates a batch record.
2. **Parse and chunk:** applies the prototype's cleaning, claim re-assembly and chunking, and writes
   the results to `processed/`.
3. **Embed:** computes chunk vectors on GPU Spot instances during the backfill and on CPU for weekly
   files, and writes them to `embeddings/`.
4. **Load:** writes patent records, claims and chunk text to PostgreSQL.
5. **Index:** bulk-indexes patents and chunks into OpenSearch.
6. **Verify:** runs count checks and canary queries, and then makes the batch visible to queries.

**Query processing (blue arrows).**

- **A.** An examiner submits a query with filters to the Search API.
- **B.** The API embeds the query, resolves keyword filters and runs a filtered ANN search in
  OpenSearch.
- **C.** The API retrieves titles and evidence text from PostgreSQL and returns the ranked patents.

Dashed arrows denote the schedule trigger and metrics. The diagram is generated from
[`diagrams/architecture.py`](diagrams/architecture.py) with the
[`diagrams`](https://diagrams.mingrammer.com) library
(`uv run --group docs python docs/diagrams/architecture.py`; Graphviz is required).

### 5.2 Components

| Component | Technology | Responsibility | Rationale |
|---|---|---|---|
| Object storage | Amazon S3 | Raw files as received, processed Parquet files and embeddings per model version | Low cost and high durability. Any downstream store can be rebuilt without re-parsing or re-embedding |
| System of record and job queue | Amazon RDS for PostgreSQL | Patents, claims, chunk text and aliases; the `batches`, `jobs` and `index_versions` tables | Transactions support idempotent upserts, and `SELECT … FOR UPDATE SKIP LOCKED` provides a job queue without an additional service |
| Search | Amazon OpenSearch Service with the k-NN plugin | `patents` index: title and abstract text, CPC codes, dates and normalised titles. `chunks` index: one document per chunk with its vector and copied filter fields | Keyword search (BM25), structured filters and ANN search in one engine. Filters are evaluated during the ANN search ("efficient filtering"), and exact search is used when the filtered set is small, which mirrors the cost-based selection of the prototype (`choose_path`) |
| Ingestion workers | AWS Batch (containers on Spot instances; GPU instances for the backfill embedding) | Execution of the pipeline stages | Stateless, and scaled to zero between weekly runs |
| Scheduler | Amazon EventBridge | Start of the weekly ingestion run | Managed schedule with no server to operate |
| Search API | The prototype's FastAPI application on Amazon ECS (Fargate), CPU only | Query embedding (ONNX bge-small), query planning, grouping and result assembly | Retains the prototype's response format, so the web interface and the CLI work unchanged |
| Monitoring | Amazon CloudWatch (optionally Grafana) | Pipeline and query metrics, alarms | See Section 11 |

### 5.3 Reuse of Prototype Code

| Prototype module | Role in the scaled system |
|---|---|
| `load.py`, `normalize.py`, `claims.py`, `chunk.py`, `tokens.py` | Used unchanged by the parse stage, one batch at a time; the loader already reads files incrementally |
| `embed.py` | Used by the embed stage (GPU) and by the API, exported to ONNX for CPU inference |
| `filters.py` (`FilterSpec`, `compile_filter`) | Interface unchanged; a translator to the OpenSearch query DSL is added alongside `filter_sql` |
| `vindex.py` (`VectorIndex` protocol) | A new `OpenSearchIndex` class implements `top_docs`, `top_rows` and `doc_row_scores` |
| `search.py` (`SearchEngine`, `SearchResponse`) | Orchestration and response types unchanged; storage calls are directed to OpenSearch and PostgreSQL |
| `store.py` (SQLite) | Replaced by PostgreSQL and S3 |

## 6. Data Model

### 6.1 Object Storage Layout

All keys are deterministic, so re-running a stage overwrites its previous output instead of creating
duplicates.

```text
s3://patsearch/raw/<pub_week>/<source_file>                     source file as received; never modified
s3://patsearch/processed/<batch_id>/patents.parquet             cleaned patents (prototype Patent fields and flags)
s3://patsearch/processed/<batch_id>/chunks.parquet              chunk_id, doc_number, kind, label, embed_text, chunk_hash
s3://patsearch/processed/<batch_id>/rejects.jsonl               dropped records with the reason
s3://patsearch/embeddings/<model_version>/<batch_id>.parquet    chunk_id, chunk_hash, vector (float16)
```

`chunk_id` has the form `<doc_number>:<ordinal>`. `chunk_hash` is the SHA-1 hash of the embedding
text and serves as the key of the embedding cache, so a chunk whose text has not changed is not
embedded again.

### 6.2 Relational Schema

The following abbreviated DDL defines the PostgreSQL tables.

```sql
CREATE TABLE batches (
  batch_id      bigserial PRIMARY KEY,
  source_uri    text NOT NULL,
  sha256        text NOT NULL UNIQUE,       -- registering the same file twice has no effect
  pub_week      date,
  state         text NOT NULL DEFAULT 'registered',
                -- registered | parsed | embedded | loaded | indexed | verified | needs_review | failed
  stats         jsonb,                      -- load statistics, claim and chunk counts, per-stage timings
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
  not_before   timestamptz DEFAULT now(),   -- earliest retry time (backoff)
  worker       text, heartbeat_at timestamptz, last_error text,
  UNIQUE (batch_id, stage)
);
CREATE TABLE patents (
  doc_number    text PRIMARY KEY,
  canonical     text REFERENCES patents,    -- for aliases: the canonical publication (an A9 points to its A1)
  kind text, title text, abstract text, pub_date date,
  cpc_codes     text[],                     -- the source data can list several codes per patent
  has_description boolean, flags jsonb,
  content_hash  text,                       -- deduplication key from the prototype
  batch_id      bigint REFERENCES batches,
  version       int NOT NULL DEFAULT 1,     -- incremented when a corrected publication replaces the record
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

Workers claim jobs with the following statement. `SKIP LOCKED` ensures that concurrent workers never
claim the same job.

```sql
UPDATE jobs SET status = 'running', attempts = attempts + 1, worker = $1, heartbeat_at = now()
WHERE job_id = (SELECT job_id FROM jobs
                WHERE status = 'queued' AND stage = $2 AND not_before <= now()
                ORDER BY job_id FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING *;
```

### 6.3 Search Indexes

The mappings below outline both OpenSearch indexes. Field options must be verified against the
deployed version; on-disk compressed vectors require OpenSearch 2.17 or later.

```jsonc
// patents index: 10 million documents, approximately 4 primary shards.
// Text analyzer: standard tokenizer, lowercase and ASCII folding, no stemming
// (equivalent to the prototype's FTS5 behaviour).
{ "doc_number": "keyword", "title": "text", "abstract": "text",
  "title_norm": "keyword", "cpc_codes": "keyword", "cpc_groups": "keyword",
  "cpc_subclasses": "keyword", "pub_date": "date", "batch_id": "integer" }

// chunks index: 378 million documents, 18–24 primary shards of 20–40 GB each.
// _source excludes the vector.
{ "chunk_id": "keyword", "doc_number": "keyword", "kind": "byte", "batch_id": "integer",
  "cpc_codes": "keyword", "cpc_groups": "keyword", "cpc_subclasses": "keyword", "pub_date": "date",
  "vector": { "type": "knn_vector", "dimension": 384, "space_type": "innerproduct",
              "mode": "on_disk", "compression_level": "32x" } }
// 32x compression: 1 bit per dimension in memory; full-precision vectors on disk for rescoring.
```

The CPC and date fields are small, so they are copied onto every chunk, which allows them to be
evaluated inside the ANN search. Title and abstract text is not copied, because that would index
the same text 38 times per patent. Keyword filters are instead resolved on the `patents` index
(Section 8).

## 7. Ingestion Pipeline

### 7.1 Stages

A batch corresponds to one weekly source file. For the backfill, the corpus is divided into batches
of approximately 10,000 patents, which gives about 1,000 batches. Each stage reads the output of the
previous stage from S3, writes its own output to a deterministic location and enqueues the job for
the next stage. Any stage can therefore be re-run without side effects.

| # | Stage | Processing | Output | Compute | Failure handling |
|---|---|---|---|---|---|
| 1 | Register | Copies the new file to `raw/`, computes its SHA-256 checksum and inserts a `batches` record | `batches` record | Negligible | A duplicate checksum means the file was already ingested, and the stage ends without changes |
| 2 | Parse | Runs the prototype's `load`, `dedupe`, `segment_claims` and `chunk_patent` | Parquet files and `rejects.jsonl` under `processed/<batch_id>/`; batch statistics | About 12 ms per patent per core | Invalid records are written to `rejects.jsonl` and the batch continues. A drop rate above 1% sets the batch to `needs_review` |
| 3 | Embed | Embeds the chunks that are not yet in the cache for `(model_version, chunk_hash)` | `embeddings/<model_version>/<batch_id>.parquet` | GPU (backfill), CPU (weekly) | After a GPU out-of-memory error, the batch size is halved and the stage is retried. The output is checked for completeness, NaN values and unit norm |
| 4 | Load | Upserts patents, claims and chunks in one transaction per batch, deduplicates across batches on `content_hash` (A9 republications become aliases), and increments `version` for corrected publications | PostgreSQL rows | CPU, minutes | The transaction is rolled back as a whole and retried |
| 5 | Index | Bulk-indexes into `patents_vN` and `chunks_vN`, using the document number and the chunk ID as document IDs | OpenSearch documents | CPU | Only items reported as failed by the bulk API are retried. Chunks of republished patents are removed with delete-by-query |
| 6 | Verify | Checks the document counts of the batch and runs canary queries: for 20 random patents of the batch, claim 1 must return its own patent at rank 1 | Batch state `verified` | Seconds | On failure, the batch is set to `needs_review` and remains invisible to queries |

### 7.2 Visibility of New Data

Every document carries its `batch_id`. Queries are implicitly restricted to verified batches through
a cached set of batch IDs, so partially indexed batches are never served. Each response reports
`index_as_of`, the publication week of the most recent verified batch.

### 7.3 Backfill and Weekly Operation

For the backfill, the indexes are created with `number_of_replicas: 0` and `refresh_interval: -1`,
and approximately 10 GPU Spot workers process the corpus in about 11 hours (Section 9.1). Replicas
and refresh are enabled once the backfill is complete. Weekly operation uses the same pipeline on a
single file of roughly 60 million tokens, which can be embedded on CPU.

### 7.4 Model Changes

A change of embedding model is deployed as a blue-green re-index. All chunks are embedded under a new
`embeddings/<model_version>/` prefix, new indexes (`patents_vN+1` and `chunks_vN+1`) are built and
evaluated, and the `patents` and `chunks` aliases are switched atomically. The previous indexes are
retained for one week to allow rollback. Vectors produced by different models are never stored in
the same index; the prototype already enforces this rule in `SearchEngine.embedder`.

## 8. Query Processing

The system supports the same three query types and returns the same response format as the
prototype. A text query is processed in the following steps:

1. **Filter validation.** The prototype's `compile_filter` validates the filters; invalid input
   returns HTTP 400.
2. **Query vector.** Text queries are embedded with ONNX bge-small on CPU in about 10 ms, with an LRU
   cache keyed by the query string. Patent and claim queries use stored vectors and require no model
   inference.
3. **Filter resolution.** CPC and date constraints are passed directly to the ANN search, because
   these fields are present on every chunk. Keyword, exact-title and title-phrase constraints are
   first evaluated on the `patents` index:
   - If no patent matches, an empty result is returned.
   - If at most 65,536 patents match (the OpenSearch `terms` limit), their document numbers are
     added to the ANN filter.
   - If more patents match, the filter is not pushed down. The ANN request is enlarged to
     k′ = min(10,000, 3k / selectivity) candidates, and chunks whose patent fails the filter are
     discarded. Because such filters are broad, most candidates pass.

   The choice between pre-filtering and post-filtering is thus made per query from the measured
   selectivity, as in the prototype.
4. **ANN search.** The filtered ANN search over `chunks` retrieves 20 × k chunks. OpenSearch performs
   an exact search when the filtered set is small and a filtered HNSW search otherwise, and it
   re-scores the 1-bit candidates with full-precision vectors.
5. **Grouping.** The first hit for each document number is that patent's best chunk and determines
   its rank. If fewer than k patents are returned, the search is repeated with a larger number of
   chunks.
6. **Evidence.** A second ANN search restricted to the top k patents (about 380 chunks, evaluated
   exactly) selects the best claim, paragraph and abstract for each result.
7. **Result assembly.** Titles and evidence text are read from PostgreSQL by primary key.

For similar-patent queries, one ANN request is issued for each source vector (the abstract and the
independent claims, up to eight vectors in total, sent as a single multi-search request). The
candidate patents are combined, and each is scored by the mean of its best match per source vector,
following the prototype's max-sim method. Claim-mapping queries use the stored claim vector and
follow steps 3 to 7.

| Step | Expected latency |
|---|---|
| Query embedding (text queries only, CPU) | 5–15 ms |
| Keyword filter resolution on `patents` (if present) | 5–30 ms |
| Filtered ANN search and rescoring on `chunks` | 20–100 ms |
| Evidence search (top k patents only) | 5–20 ms |
| Result assembly from PostgreSQL | 5–10 ms |
| **Total** | **40–180 ms** (95th-percentile target: 500 ms) |

## 9. Cost Estimate

Prices are approximate AWS on-demand list prices for the us-east-1 region, with an uncertainty of
about ±30%, and should be confirmed with the AWS Pricing Calculator. Reserved instances reduce
compute costs by 30–40%.

### 9.1 One-Time Backfill

| Item | Estimate (USD) | Basis |
|---|---|---|
| Parsing, cleaning and chunking | 5 | 34 CPU-hours on Spot instances |
| Embedding with self-hosted bge-small | 30–200 | 59 billion tokens at an estimated 150,000 tokens per second per NVIDIA T4 GPU (about five times the measured M4 throughput), which is about 110 GPU-hours, or 11 hours on 10 Spot GPUs at USD 0.20–0.50 per hour. The upper bound assumes a throughput three times lower |
| Embedding through a hosted API (alternative) | 1,200 | At about USD 0.02 per million tokens; this would also replace the prototype's embedding model |
| Bulk loading of OpenSearch and PostgreSQL | Included in the monthly cost | Indexing takes between several hours and one day on the cluster |

### 9.2 Monthly Operation

| Item | Estimate (USD per month) | Basis |
|---|---|---|
| OpenSearch data nodes | 1,500–2,900 | About 70 GB of k-NN memory (378 million vectors × (48 bytes per 1-bit vector + about 128 bytes of HNSW links)), served by three nodes with 64 GB of memory, or 128 GB for additional headroom |
| OpenSearch master nodes and storage | 450 | Three small dedicated master nodes and about 1.5 TB of gp3 storage for the full-precision vectors used in rescoring, with one replica |
| PostgreSQL (RDS, 16 GB memory, 500 GB gp3, backups) | 270 | Text, metadata and pipeline state only; no vectors |
| Search API (two small CPU containers) | 110 | |
| Amazon S3 (about 650 GB: raw files, Parquet files, float16 embeddings) | 15 | |
| Weekly pipeline workers | 10 | About one CPU-hour per week |
| Monitoring, logging and other services | 100 | |
| **Total** | **2,500–4,000** | |

### 9.3 Reduced-Cost Configuration

A configuration based only on PostgreSQL and pgvector costs approximately USD 1,000 per month. It
maintains one HNSW index over 10–30 million summary vectors and re-scores the chunks of the top N
candidates exactly, using chunk vectors stored as packed bytes per patent. Its recall is lower
(Section 4), which makes it suitable for demonstrations but not for prior-art search.

### 9.4 Further Savings

- Reducing the number of vectors, for example by merging description paragraphs into chunks of
  about 500 tokens, which yields approximately 30% fewer vectors. Recall must be re-measured after
  such a change.
- Purchasing reserved instances.
- Using an IVF-PQ index instead of HNSW.
- Operating without a replica, at the cost of availability.

## 10. Error Handling

| Area | Failure | Handling |
|---|---|---|
| Records | Missing, null or incorrectly typed fields; claims that cannot be parsed | The prototype's tolerant loader substitutes empty values and records a flag. A record is dropped only if it has no document number or no text, and rejected records are written to S3 together with the reason |
| Batch quality | A drop rate above 1%, or an unusual change in the distribution of data-quality flags (for example, a sharp fall in the share of fully reconstructed claims) | The batch is set to `needs_review`, is not served and raises an alert. This check detects upstream format changes such as the claim fragmentation found in the sample data |
| Stage execution | Transient errors from the network, S3 or the database, and Spot interruptions | The job is retried with exponential backoff (`not_before`) up to three times and is then marked `failed`, which raises an alert. A reaper process re-queues running jobs whose heartbeat is older than ten minutes |
| Embedding | Out-of-memory errors; NaN values | The batch size is halved and the stage is retried. NaN values or vectors without unit norm cause the stage to fail |
| Indexing | Partial bulk failures; mapping errors | Only the failed items are retried. Mapping errors cause the stage to fail, because they indicate a code defect rather than a data problem |
| Cross-store consistency | Divergence between PostgreSQL and OpenSearch, since there are no distributed transactions | Upserts are idempotent and keyed by document number and chunk ID. A nightly reconciliation job compares per-batch counts and `content_hash` values across PostgreSQL and both indexes, and re-queues the index stage where they differ |
| Query input | An invalid CPC code, date or scope | HTTP 400 with the prototype's error message |
| Query dependencies | OpenSearch is slow or unavailable; the embedding model is unavailable | The request times out and is retried once with a smaller k; otherwise the API returns HTTP 503 with an explanatory message. Patent and claim queries remain available without the model, because they use stored vectors |
| Partially ingested data | A batch has not yet been verified | The batch is not visible to queries, and every response states `index_as_of` |

## 11. Monitoring and Status Tracking

- **Pipeline status.** The `batches` and `jobs` tables are the authoritative record of pipeline
  status. A dashboard shows the number of batches in each state, the oldest unverified batch, failed
  jobs with their `last_error`, the throughput of each stage (patents and tokens per second) and the
  estimated completion time of the backfill.
- **Corpus contents.** Patent counts per CPC subclass and per publication week reveal gaps such as a
  missing weekly file. Per-batch data-quality metrics in `stats` (drop rate, share of patents without
  a description, share of fully reconstructed claims, truncated chunks) are already computed by the
  prototype's build. The `index_versions` table records which model and index are live, together with
  their document counts and evaluation scores.
- **Search health.** Median and 95th-percentile latency per query type, queries per second, error
  rate, share of queries without results, distribution of filter selectivity and cache hit rate.
- **Relevance canaries.** A fixed evaluation set runs nightly. It consists of known-item tests (a
  patent's own claim must return that patent at rank 1) and the prototype's benchmark queries with
  their expected top results. Recall@10 is tracked for each index version.
- **Alerts.** Alerts are raised when a job is `failed`, a batch is set to `needs_review`, ingestion
  lags by more than eight days, PostgreSQL and OpenSearch counts differ, the 95th-percentile query
  latency exceeds 500 ms for ten minutes, or canary recall falls by more than five percentage points.

## 12. Risks and Open Challenges

1. **Vector memory.** The 378 million vectors fit in memory only with 1-bit compression, which
   reduces recall. Rescoring with full-precision vectors from disk must compensate for this, and
   oversampling and `ef_search` must be tuned against the canary set. The number of vectors is the
   main cost driver.
2. **Correctness of filtered ANN search.** Applying filters after the ANN search returns too few
   results: in the prototype benchmark, post-filtering the top 100 results reached a recall of only
   0.06–0.38 for selective filters. Filters must therefore be evaluated inside the ANN search. Broad
   keyword filters, which cannot be pushed down, remain the weakest case.
3. **Keyword filters across two indexes.** Title and abstract text is held in the `patents` index
   and the vectors in the `chunks` index. Because the push-down of document numbers is limited to
   65,536 values, very broad keyword filters rely on oversampling. Possible remedies are cached
   bitsets for frequent filters and a compact keyword field on the chunk documents.
4. **Re-indexing cost and duration.** A model change requires re-embedding 59 billion tokens and
   rebuilding the indexes, which takes between several hours and one day and temporarily doubles the
   storage requirement. The embedding cache and the blue-green aliases keep this operation
   predictable and reversible.
5. **Heterogeneous source data.** Production data contains multiple CPC codes per patent, additional
   sources and XML formats, non-English text, corrected publications and continuations. The tolerant
   loader and the per-batch quality checks handle most of these cases; deduplication by content hash
   covers only exact republications.
6. **Absence of relevance labels.** Without an evaluation set, quantization and ANN parameters cannot
   be tuned reliably. An evaluation set should be built early, for example from examiner citations
   and known-item tests.
7. **Consistency across S3, PostgreSQL and OpenSearch.** In the absence of distributed transactions,
   consistency relies on deterministic identifiers, idempotent stages, the verified-batch gate and
   nightly reconciliation.
