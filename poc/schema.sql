-- Subset of the Part 2 design schema (docs/part2_system_design.md, Section 6.2).
-- Loaded automatically by the Postgres container on first start.

CREATE TABLE batches (
  batch_id      bigserial PRIMARY KEY,
  source_uri    text NOT NULL,
  sha256        text NOT NULL UNIQUE,              -- re-registering the same file has no effect
  pub_week      date,
  state         text NOT NULL DEFAULT 'registered',
                -- registered | parsed | embedded | loaded | indexed | verified | needs_review | failed
  stats         jsonb NOT NULL DEFAULT '{}',
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE jobs (
  job_id        bigserial PRIMARY KEY,
  batch_id      bigint NOT NULL REFERENCES batches,
  stage         text NOT NULL,                      -- parse | embed | load | index | verify
  status        text NOT NULL DEFAULT 'queued',     -- queued | running | done | failed
  attempts      int  NOT NULL DEFAULT 0,
  not_before    timestamptz NOT NULL DEFAULT now(), -- earliest (re)try time
  worker        text,
  heartbeat_at  timestamptz,
  started_at    timestamptz,
  finished_at   timestamptz,
  last_error    text,
  UNIQUE (batch_id, stage)
);
CREATE INDEX jobs_claimable ON jobs (status, not_before, job_id);

CREATE TABLE patents (
  doc_number    text PRIMARY KEY,
  title         text,
  abstract      text,
  pub_date      date,
  cpc           text,
  content_hash  text,
  batch_id      bigint REFERENCES batches,
  version       int NOT NULL DEFAULT 1,
  updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  chunk_id      text PRIMARY KEY,                   -- <doc_number>:<ordinal>
  doc_number    text NOT NULL REFERENCES patents,
  batch_id      bigint NOT NULL REFERENCES batches,
  kind          text NOT NULL,
  label         text NOT NULL,
  text          text NOT NULL
);
CREATE INDEX chunks_batch ON chunks (batch_id);
