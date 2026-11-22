CREATE EXTENSION IF NOT EXISTS vector;

-- Embedding dim 384 = sentence-transformers/all-MiniLM-L6-v2 (free, runs locally).
-- If you use OpenAI text-embedding-3-small, change to vector(1536).

CREATE TABLE ingestion_runs (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    jobs_seen     INT NOT NULL DEFAULT 0,
    jobs_inserted INT NOT NULL DEFAULT 0,
    jobs_updated  INT NOT NULL DEFAULT 0,
    error         TEXT
);

CREATE TABLE jobs (
    id           BIGSERIAL PRIMARY KEY,
    source       TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    title        TEXT NOT NULL,
    company      TEXT,
    location     TEXT,
    remote       BOOLEAN NOT NULL DEFAULT FALSE,
    description  TEXT NOT NULL,
    url          TEXT,
    posted_at    TIMESTAMPTZ,
    content_hash TEXT NOT NULL,          -- skip re-embedding unchanged jobs
    embedding    vector(384),
    -- Precomputed full-text document: title matches (A) outrank description matches (B).
    -- Postgres keeps it in sync automatically, so ingestion code does not touch it.
    fts          tsvector GENERATED ALWAYS AS (
                     setweight(to_tsvector('english', title), 'A') ||
                     setweight(to_tsvector('english', description), 'B')
                 ) STORED,
    run_id       BIGINT REFERENCES ingestion_runs(id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, external_id)         -- dedupe key for upserts
);

CREATE TABLE saved_jobs (
    id        BIGSERIAL PRIMARY KEY,
    user_ref  TEXT NOT NULL,             -- session/user id, no accounts needed for MVP
    job_id    BIGINT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    saved_at