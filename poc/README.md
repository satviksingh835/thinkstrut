# Part 2 Proof of Concept: Ingestion Pipeline and Job Queue

This proof of concept implements the ingestion pipeline of the Part 2 design
([`docs/part2_system_design.md`](../docs/part2_system_design.md), Sections 6.2, 7, 10 and 11) on a
single machine with Docker. A PostgreSQL job queue drives several worker containers through the six
pipeline stages, using the Part 1 code to process the 64 sample weekly files, and a dashboard shows
the status of every batch.

## Components

| Container | Stands in for | Purpose |
|---|---|---|
| `db` (PostgreSQL 16) | Amazon RDS for PostgreSQL | `batches`, `jobs`, `patents` and `chunks` tables ([`schema.sql`](schema.sql)) |
| `scheduler` | Amazon EventBridge and stage 1 | Registers one weekly file per second, plus one corrupt file |
| `worker` (3 replicas) | AWS Batch jobs | Claim jobs with `FOR UPDATE SKIP LOCKED` and run stages 2–6 |
| `dashboard` | Amazon CloudWatch dashboard | Batch states, stage progress, workers, errors and recoveries |
| `lake` volume | Amazon S3 | `raw/`, `processed/`, `embeddings/` and `index/` |

Code: [`pipeline/db.py`](pipeline/db.py) (job queue: claim, complete, retry, review, reaper,
heartbeat), [`pipeline/stages.py`](pipeline/stages.py) (the stages, reusing `patsearch/`),
[`pipeline/worker.py`](pipeline/worker.py), [`pipeline/scheduler.py`](pipeline/scheduler.py) and
[`pipeline/dashboard.py`](pipeline/dashboard.py).

## Running

Docker must be running, and the sample data must be in `data/patent_data_small/`.

```bash
docker compose -f poc/docker-compose.yml up --build -d   # start everything
open http://localhost:8080                               # dashboard (JSON: /api/status)
docker compose -f poc/docker-compose.yml logs -f worker  # follow the workers
docker compose -f poc/docker-compose.yml down -v         # stop and delete all data
```

## Fault Injection

The workers inject failures so that the recovery paths of the design can be observed:

| Setting | Default | Effect | Expected recovery |
|---|---|---|---|
| `FAIL_RATE` | 0.15 | A job raises a transient error (simulated Spot interruption) | Retried with exponential backoff, up to three attempts |
| `CRASH_RATE` | 0.02 | The worker process exits in the middle of a job | Docker restarts the container; the reaper re-queues the job once its heartbeat is 15 s old |
| `BAD_FILE` | 1 | The scheduler registers a truncated JSON file | The parse stage sets the batch to `needs_review`; it is not retried |
| `STAGE_DELAY_S` | 0.6 | Simulated processing time per stage | Makes progress visible on the dashboard |

## Results

One run with the default settings (3 workers, 65 registered files):

| Measure | Result |
|---|---|
| Batches verified | 64 of 64 valid files |
| Batches in `needs_review` | 1 (the corrupt file: "source file could not be read") |
| Jobs completed | 320 (64 batches × 5 stages) |
| Transient errors recovered by retry | 71 |
| Jobs recovered by the reaper after a worker crash | 10 |
| Patents loaded | 636, identical to the Part 1 build (640 records, 4 A9 republications merged) |
| Chunks loaded | 20,872, identical to the Part 1 build with the same tokenizer |
| Total time | 97 s |

The 4 A9 republications were published in the week of 2024-07-11, eleven weeks after their A1
originals, so the deduplication was performed across batches by the load stage, as specified in the
design. Re-running the scheduler registers no new batches, because each file is identified by its
SHA-256 checksum.

## Simplifications

- A shared Docker volume replaces S3, and a NumPy file per batch replaces the OpenSearch index.
- The embed stage uses the model-free `HashEmbedder` from Part 1, which keeps the worker image small
  (no PyTorch). The stage contract is unchanged, so the production model can be substituted.
- All workers process all stages; in production, GPU workers would claim only embed jobs (the claim
  statement in the design filters by stage).
- The verify canary checks the integrity of the batch's own index rather than querying the full
  production index.
