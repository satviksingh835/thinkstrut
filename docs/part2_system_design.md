# Part 2: System Design for Patent Search at 10 Million Documents

## 1. Summary

This document describes how the Part 1 search engine (the prototype) scales to 10 million patents.
Search behaviour is unchanged: patents are split into chunks, each chunk is embedded, a patent is
ranked by its best-matching chunk, and filters are applied before vector scoring. The single-machine
components are replaced by managed AWS services: **Amazon S3** for files and embeddings, **Amazon
RDS for PostgreSQL** for records, text and pipeline state, and **Amazon OpenSearch Service** for
keyword, filter and vector search over 378 million chunk vectors compressed to one bit per dimension.
A six-stage batch pipeline ingests one weekly file at a time.

Estimated cost: USD 50–200 for the one-time backfill and USD 2,500–4,000 per month. A reduced
configuration costs about USD 1,000 per month, with lower recall (Section 9.3).

## 2. Scope and Assumptions

- A corpus of 10 million patents, growing by about 10,000 per week (the USPTO publishes roughly
  7,000 applications per week).
- At most 20 queries per second; latency targets of 150 ms (median) and 500 ms (95th percentile).
- The prototype's three query types (text, similar patents, claim mapping), hybrid filters and
  response format.

## 3. Capacity Estimates

Measured per patent on the prototype index (636 patents) and scaled linearly:

| Quantity | Per patent | At 10 million patents |
|---|---|---|
| Chunks | 37.8 | 378 million |
| Tokens to embed | 5,933 | 59 billion |
| Raw JSON | 24.9 KB | 250 GB |
| Text, metadata and keyword index | 37.7 KB | 380 GB |
| Chunk vectors (384 dimensions): float32 / float16 / 1-bit | 58 KB / 29 KB / 1.8 KB | 581 GB / 290 GB / 18 GB |
| CPU time for parsing and chunking | 12.4 ms | 34 CPU-hours |
| Embedding time on an Apple M4 GPU (31,900 tokens/s) | 0.19 s | 516 hours |

The chunk vectors determine the design: exhaustive search would read 145–581 GB per query, and an
uncompressed ANN index would need 340–630 GB of memory.

## 4. Design Decision: Index Granularity

- **Option A:** index one or a few summary vectors per patent (10–30 million vectors, which fit in
  PostgreSQL with pgvector), then re-score the chunks of the top N candidates exactly.
- **Option B:** index every chunk, as the prototype does, and compress the vectors.

Recall@10 of Option A against the prototype's exact search (20 benchmark queries):

| Summary vectors per patent | N = 10 | N = 20 | N = 50 | N = 100 |
|---|---|---|---|---|
| 1: mean of abstract and claim chunks | 0.585 | 0.790 | 0.880 | 0.920 |
| 1: mean of all chunks | 0.645 | 0.815 | 0.900 | 0.955 |
| 3: abstract, claims and description (maximum score) | 0.620 | 0.805 | 0.910 | 0.965 |

Even with 100 candidates, 16% of the test corpus, Option A misses 4–8% of the correct results; at
10 million patents, 1,000 candidates cover only 0.01% of the corpus. A summary vector averages the
whole patent, while queries usually match a single claim or paragraph. Option B is adopted, and
Option A is kept as the reduced-cost configuration (Section 9.3).

## 5. Architecture

### 5.1 Overview

![Architecture: inside the AWS Cloud boundary, six AWS Batch ingestion jobs write to Amazon S3, Amazon RDS for PostgreSQL and Amazon OpenSearch Service; the Search API on ECS Fargate serves patent examiners from OpenSearch and RDS; EventBridge triggers ingestion and CloudWatch collects metrics](images/architecture.png)

| Element in the diagram | Type | Role in the system |
|---|---|---|
| USPTO bulk data (weekly files) | External data source | Publishes new patent applications as weekly bulk files; the only input of the indexing pipeline |
| AWS Cloud (us-east-1) | Deployment boundary | Encloses every component operated by the system; the data source and the users are external |
| Amazon EventBridge (weekly trigger) | Scheduler | Starts the indexing pipeline once a week by creating the job for step 1 |
| Ingestion group: AWS Batch jobs (Spot) | Compute | Runs the six indexing stages as containerised batch jobs on Spot instances |
| 1 Register | Indexing stage | Copies a new weekly file to S3 and creates a batch record in PostgreSQL |
| 2 Parse & chunk | Indexing stage | Cleans the records, re-assembles the claims and splits each patent into chunks |
| 3 Embed (GPU) | Indexing stage | Converts each chunk into a 384-dimensional vector |
| 4 Load | Indexing stage | Writes patent records, claims and chunk text to PostgreSQL |
| 5 Index | Indexing stage | Writes patent documents and chunk vectors to the OpenSearch indexes |
| 6 Verify | Indexing stage | Checks document counts and runs canary queries before the batch becomes searchable |
| Storage group | Data layer | The three data stores shared by both pipelines: written by the indexing pipeline, read by the search pipeline |
| Amazon S3 (raw / processed / embeddings) | Object storage | Keeps the raw files, intermediate Parquet files and embeddings, so that any stage can be re-run and the indexes rebuilt |
| Amazon RDS for PostgreSQL | Relational database | System of record for patent records and text; also holds the job queue and the status of every batch |
| Amazon OpenSearch Service | Search engine | Holds the `patents` index (keyword search and filters) and the `chunks` index (vectors and filters), and executes searches |
| Serving group: Amazon ECS on Fargate | Compute | Runs the Search API as serverless containers |
| Search API (FastAPI) | Application service | Receives queries, converts them into vectors, queries OpenSearch and PostgreSQL, and returns ranked patents |
| Amazon CloudWatch (metrics & alarms) | Monitoring | Collects metrics from the pipeline and the API, and raises alarms |
| Patent examiners (web UI / CLI) | Users | Submit searches and review the results |
| Dark solid arrows, steps 1–6 | Indexing data flow | Movement of data through the indexing stages and into the data stores |
| Blue arrows, steps A–C | Search data flow | Path of a search request through the system |
| Dashed arrows | Control and monitoring | Schedule trigger and metrics |

The system consists of two pipelines that share the storage layer: an indexing pipeline, which turns
weekly USPTO files into searchable indexes, and a search pipeline, which answers queries from patent
examiners.

#### Indexing pipeline

The indexing pipeline runs as six AWS Batch jobs. The jobs coordinate through the job table in
PostgreSQL: each stage claims a job, processes one batch and then enqueues the job for the next
stage.

1. **Register.** Once a week, EventBridge triggers the Register stage, which copies the new file
   from the USPTO to S3 and records a new batch in PostgreSQL.
2. **Parse and chunk.** The stage reads the raw file, cleans the records, re-assembles the claims and
   splits each patent into chunks, and writes the result to S3.
3. **Embed.** The stage converts each chunk into a vector, reuses cached vectors for text that has not
   changed, and stores the vectors in S3.
4. **Load.** The stage writes patent records, claims and chunk text to PostgreSQL, which later
   supplies the text shown in search results.
5. **Index.** The stage writes the patent documents and chunk vectors to OpenSearch.
6. **Verify.** The stage checks document counts and runs canary queries. Only after this check
   succeeds does the batch become visible to searches.

Because every stage reads its input from S3 and writes deterministic output, a failed stage can be
retried without repeating earlier work. The initial backfill of 10 million patents runs the same
stages over approximately 1,000 batches. Section 7 describes the stages in detail.

#### Search pipeline

The search pipeline reads the data produced by the indexing pipeline and does not modify it.

- **A. Request.** An examiner submits a text query, a patent number or a claim, together with
  optional filters, to the Search API.
- **B. Retrieval.** The API converts the query into a vector, or loads the stored vector of the given
  patent or claim. It then resolves keyword filters on the `patents` index and runs a filtered ANN
  search over the `chunks` index in OpenSearch. The best-matching chunk of each patent determines the
  patent's rank.
- **C. Result assembly.** The API retrieves titles and evidence text for the top results from
  PostgreSQL and returns the ranked patents, each with the passages that matched.

Section 8 describes query processing in detail.

#### Interaction between the pipelines

The two pipelines are independent apart from the shared storage layer: the indexing pipeline
writes, and the search pipeline reads. A new batch becomes searchable only after step 6 marks it as
verified, so searches never see partially indexed data. Both pipelines report metrics to CloudWatch,
which raises alarms on failed jobs, ingestion delays and slow queries.

The diagram is generated from [`diagrams/architecture.py`](diagrams/architecture.py) with the
[`diagrams`](https://diagrams.mingrammer.com) library
(`uv run --group docs python docs/diagrams/architecture.py`; Graphviz is required).

### 5.2 Components

| Component | Technology | Responsibility | Rationale |
|---|---|---|---|
| Object storage | Amazon S3 | Raw files, processed Parquet files, embeddings | Durable and inexpensive; any store can be rebuilt from it |
| Database and job queue | Amazon RDS for PostgreSQL | Records, claims, chunk text, aliases; job and batch tables | Transactional upserts; `SKIP LOCKED` provides a job queue without a separate service |
| Search | Amazon OpenSearch Service (k-NN plugin) | `patents` index (text, CPC codes, dates); `chunks` index (vectors and filter fields) | Keyword, filter and vector search in one engine, with filters applied inside the ANN search |
| Ingestion workers | AWS Batch on Spot instances (GPU for the backfill embedding) | Pipeline stages | Stateless; scale to zero between runs |
| Scheduler | Amazon EventBridge | Weekly trigger | Managed; no server to operate |
| Search API | Prototype FastAPI application on Amazon ECS (Fargate) | Query embedding, query planning, result assembly | Keeps the prototype's response format |
| Monitoring | Amazon CloudWatch | Metrics and alarms | See Section 11 |

### 5.3 Reuse of Prototype Code

| Prototype module | Use in the scaled system |
|---|---|
| `load.py`, `normalize.py`, `claims.py`, `chunk.py`, `tokens.py` | Parse stage, unchanged |
| `embed.py` | Embed stage; API (exported to ONNX for CPU) |
| `filters.py` | Unchanged interface, plus a translator to the OpenSearch query DSL |
| `vindex.py` | New `OpenSearchIndex` implementing the `VectorIndex` protocol |
| `search.py` | Unchanged orchestration and response types |
| `store.py` | Replaced by PostgreSQL and S3 |

## 6. Data Model

### 6.1 Object Storage Layout

Keys are deterministic, so re-running a stage overwrites its output instead of duplicating it.

```text
s3://patsearch/raw/<pub_week>/<source_file>                     source file as received
s3://patsearch/processed/<batch_id>/patents.parquet             cleaned patents
s3://patsearch/processed/<batch_id>/chunks.parquet              chunk_id, doc_number, kind, label, embed_text, chunk_hash
s3://patsearch/processed/<batch_id>/rejects.jsonl               dropped records with the reason
s3://patsearch/embeddings/<model_version>/<batch_id>.parquet    chunk_id, chunk_hash, vector (float16)
```

`chunk_id` is `<doc_number>:<ordinal>`. `chunk_hash` (SHA-1 of the embedding text) keys the embedding
cache, so unchanged chunks are never embedded twice.

### 6.2 Relational Schema

```sql
CREATE TABLE batches (
  batch_id      bigserial PRIMARY KEY,
  source_uri    text NOT NULL,
  sha256        text NOT NULL UNIQUE,       -- re-registering the same file has no effect
  pub_week      date,
  state         text NOT NULL DEFAULT 'registered',
                -- registered | parsed | embedded | loaded | indexed | verified | needs_review | failed
  stats         jsonb,                      -- load statistics, counts, per-stage timings
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
  not_before   timestamptz DEFAULT now(),   -- earliest retry time
  worker       text, heartbeat_at timestamptz, last_error text,
  UNIQUE (batch_id, stage)
);
CREATE TABLE patents (
  doc_number    text PRIMARY KEY,
  canonical     text REFERENCES patents,    -- for aliases: the canonical publication
  kind text, title text, abstract text, pub_date date,
  cpc_codes     text[],
  has_description boolean, flags jsonb,
  content_hash  text,                       -- deduplication key
  batch_id      bigint REFERENCES batches,
  version       int NOT NULL DEFAULT 1,     -- incremented by corrected publications
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

Workers claim jobs with the statement below; `SKIP LOCKED` prevents two workers from claiming the same
job.

```sql
UPDATE jobs SET status = 'running', attempts = attempts + 1, worker = $1, heartbeat_at = now()
WHERE job_id = (SELECT job_id FROM jobs
                WHERE status = 'queued' AND stage = $2 AND not_before <= now()
                ORDER BY job_id FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING *;
```

### 6.3 Search Indexes

On-disk compressed vectors require OpenSearch 2.17 or later; field options should be checked against
the deployed version.

```jsonc
// patents index: 10 million documents, about 4 primary shards.
// Text analyzer: standard tokenizer, lowercase, ASCII folding, no stemming.
{ "doc_number": "keyword", "title": "text", "abstract": "text",
  "title_norm": "keyword", "cpc_codes": "keyword", "cpc_groups": "keyword",
  "cpc_subclasses": "keyword", "pub_date": "date", "batch_id": "integer" }

// chunks index: 378 million documents, 18–24 primary shards of 20–40 GB; _source excludes the vector.
{ "chunk_id": "keyword", "doc_number": "keyword", "kind": "byte", "batch_id": "integer",
  "cpc_codes": "keyword", "cpc_groups": "keyword", "cpc_subclasses": "keyword", "pub_date": "date",
  "vector": { "type": "knn_vector", "dimension": 384, "space_type": "innerproduct",
              "mode": "on_disk", "compression_level": "32x" } }
// 32x: 1 bit per dimension in memory; full-precision vectors on disk for rescoring.
```

CPC and date fields are copied onto every chunk so they can be filtered inside the ANN search. Title
and abstract text is not copied; keyword filters are resolved on the `patents` index (Section 8).

## 7. Ingestion Pipeline

### 7.1 Stages

A batch is one weekly file; the backfill uses batches of about 10,000 patents (about 1,000 batches).

| # | Stage | Processing | Output | Failure handling |
|---|---|---|---|---|
| 1 | Register | Copy the file to `raw/`; record its checksum and the batch | `batches` row | Duplicate checksum: the file is skipped |
| 2 | Parse | Prototype loading, deduplication, claim re-assembly and chunking (about 12 ms per patent per core) | Parquet files, `rejects.jsonl` | Invalid records are rejected; a drop rate above 1% sets `needs_review` |
| 3 | Embed | Embed chunks missing from the cache; GPU for the backfill, CPU for weekly runs | `embeddings/` Parquet | Out-of-memory: halve the batch and retry; check count, NaN values and unit norm |
| 4 | Load | Upsert records, claims and chunks in one transaction; merge duplicates by `content_hash` | PostgreSQL rows | Roll back and retry |
| 5 | Index | Bulk-index patents and chunks, using document numbers and chunk IDs as IDs | OpenSearch documents | Retry only failed items |
| 6 | Verify | Check counts; run canary queries (claim 1 must return its own patent) | Batch `verified` | Set `needs_review`; the batch stays hidden |

### 7.2 Operation

- **Visibility:** queries only see verified batches, and each response reports `index_as_of`.
- **Backfill:** indexes are created without replicas or refresh; about 10 GPU Spot workers finish in
  about 11 hours, after which replicas and refresh are enabled.
- **Weekly runs:** one file of about 60 million tokens, embedded on CPU.
- **Model changes:** all chunks are re-embedded under a new prefix, new indexes are built and
  evaluated, and the index aliases are switched atomically. The old indexes are kept for one week for
  rollback, and vectors from different models are never mixed in one index.

## 8. Query Processing

1. **Validate filters** with `compile_filter`; invalid input returns HTTP 400.
2. **Obtain the query vector:** text is embedded with ONNX bge-small on CPU (about 10 ms, cached);
   patent and claim queries use stored vectors.
3. **Resolve filters.** CPC and date filters apply directly in the ANN search. Keyword and title
   filters are evaluated on the `patents` index first. Up to 65,536 matching patents (the OpenSearch
   `terms` limit) are passed into the ANN filter; beyond that, the ANN request is enlarged to
   k′ = min(10,000, 3k / selectivity) and non-matching results are discarded.
4. **Search** the `chunks` index for 20 × k chunks; 1-bit candidates are re-scored with
   full-precision vectors.
5. **Group** chunks by patent; each patent's first hit is its best chunk and sets its rank.
6. **Select evidence** with a second, exact search over the chunks of the top k patents.
7. **Assemble results** with titles and evidence text from PostgreSQL.

Similar-patent queries search with each source vector (the abstract and independent claims, up to
eight) and score candidates by their mean best match. Claim queries use the stored claim vector from
step 3 onwards.

| Step | Expected latency |
|---|---|
| Query embedding (text only) | 5–15 ms |
| Keyword filter resolution | 5–30 ms |
| Filtered ANN search and rescoring | 20–100 ms |
| Evidence search | 5–20 ms |
| Result assembly | 5–10 ms |
| **Total** | **40–180 ms** (target: 500 ms at the 95th percentile) |

## 9. Cost Estimate

Approximate on-demand list prices for us-east-1 (±30%); confirm with the AWS Pricing Calculator.

### 9.1 One-Time Backfill

| Item | USD | Basis |
|---|---|---|
| Parsing and chunking | 5 | 34 CPU-hours on Spot instances |
| Embedding (self-hosted bge-small) | 30–200 | 59 billion tokens at about 150,000 tokens/s per T4 GPU, or about 110 GPU-hours; the upper bound assumes a third of that throughput |
| Embedding via a hosted API (alternative) | 1,200 | About USD 0.02 per million tokens |
| Bulk loading | Included in monthly cost | Several hours to one day |

### 9.2 Monthly Operation

| Item | USD per month | Basis |
|---|---|---|
| OpenSearch data nodes | 1,500–2,900 | About 70 GB of k-NN memory: three 64 GB nodes, or 128 GB for headroom |
| OpenSearch masters and storage | 450 | Three small master nodes; 1.5 TB gp3 storage with one replica |
| PostgreSQL (RDS) | 270 | 16 GB memory, 500 GB storage, backups |
| Search API | 110 | Two small CPU containers |
| Amazon S3 | 15 | About 650 GB |
| Weekly pipeline workers | 10 | About one CPU-hour per week |
| Monitoring and other | 100 | |
| **Total** | **2,500–4,000** | |

### 9.3 Reduced-Cost Configuration

Option A (Section 4) on PostgreSQL with pgvector costs about USD 1,000 per month, but its lower recall
makes it suitable for demonstrations rather than prior-art search.

### 9.4 Further Savings

Larger description chunks (about 30% fewer vectors, with recall re-measured), reserved instances
(30–40% lower compute cost), an IVF-PQ index instead of HNSW, or operating without a replica.

## 10. Error Handling

| Area | Failure | Handling |
|---|---|---|
| Records | Missing or malformed fields | Empty value plus flag; only records without an ID or text are dropped and logged to S3 |
| Batch quality | Drop rate above 1%, or an unusual flag distribution | Batch set to `needs_review` and hidden; alert raised |
| Stage execution | Transient errors, Spot interruptions | Up to three retries with backoff, then `failed` and an alert; running jobs without a heartbeat for 10 minutes are re-queued |
| Embedding | Out-of-memory errors, NaN values | Halve the batch and retry; invalid vectors fail the stage |
| Indexing | Partial bulk failures, mapping errors | Retry failed items; mapping errors fail the stage |
| Store consistency | PostgreSQL and OpenSearch diverge | Idempotent upserts; nightly reconciliation of counts and content hashes |
| Query input | Invalid filters | HTTP 400 |
| Query dependencies | OpenSearch or the model unavailable | One retry, then HTTP 503; patent and claim queries work without the model |

## 11. Monitoring and Status Tracking

- **Pipeline:** the `batches` and `jobs` tables record the state of every batch and job; a dashboard
  shows batches per state, failures, throughput and backfill progress.
- **Contents:** patent counts per CPC subclass and week (gaps reveal missing files), per-batch
  data-quality statistics, and the live index and model in `index_versions`.
- **Search:** latency percentiles, throughput, error rate and empty-result rate.
- **Relevance:** nightly canary queries, with recall@10 tracked per index version.
- **Alerts:** failed jobs, batches in `needs_review`, ingestion lag above eight days, count
  mismatches, 95th-percentile latency above 500 ms, and recall drops above five points.

## 12. Risks and Open Challenges

1. **Vector memory.** Fitting 378 million vectors in memory requires 1-bit compression, whose recall
   loss must be recovered through rescoring and parameter tuning.
2. **Filtered search.** Post-filtering loses results (recall of 0.06–0.38 in the prototype
   benchmark), so filters must run inside the ANN search; broad keyword filters remain the weakest
   case.
3. **Keyword filters across indexes.** The 65,536-value limit forces oversampling for broad keyword
   filters.
4. **Re-indexing.** A model change requires re-embedding 59 billion tokens and temporarily doubles
   storage.
5. **Source data.** Multiple CPC codes, other formats, non-English text and corrected publications
   require extensions to the loader.
6. **Evaluation.** No relevance labels exist yet; an evaluation set, for example from examiner
   citations, is needed for tuning.
7. **Consistency.** Without distributed transactions, consistency depends on idempotent stages, the
   verification gate and reconciliation.
