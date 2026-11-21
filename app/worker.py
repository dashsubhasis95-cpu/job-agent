import httpx
from celery import Celery, chord

from .config import REDIS_URL
from .db import get_conn
from .ingest import SOURCE, fetch_page, upsert_jobs

celery_app = Celery("jobs", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_acks_late = True  # re-deliver if a worker dies mid-task


@celery_app.task(
    bind=True,
    autoretry_for=(httpx.HTTPError,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def ingest_page(self, page: int, run_id: int) -> dict:
    raw = fetch_page(page)
    seen, inserted, updated = upsert_jobs(raw, run_id)
    with get_conn() as conn:
        conn.execute(
            "UPDATE ingestion_runs SET jobs_seen = jobs_seen + %s, "
            "jobs_inserted = jobs_inserted + %s, jobs_updated = jobs_updated + %s "
            "WHERE id = %s",
            (seen, inserted, updated, run_id),
        )
    return {"page": page, "seen": seen, "inserted": inserted, "updated": updated}


@celery_app.task
def finalize_run(results: list[dict], run_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE ingestion_runs SET status = 'succeeded', finished_at = now() WHERE id = %s",
            (run_id,),
        )


@celery_app.task
def mark_failed(request, exc, traceback, run_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE ingestion_runs SET status = 'failed', finished_at = now(), error = %s "
            "WHERE id = %s",
            (str(exc)[:500], run_id),
        )


@celery_app.task
def start_ingestion(pages: int = 5) -> int:
    with get_conn() as conn:
        run_id = conn.execute(
            "INSERT INTO ingestion_runs (source) VALUES (%s) RETURNING id", (SOURCE,)
        ).fetchone()[0]
    header = [ingest_page.s(p, run_id).on_error(mark_failed.s(run_id)) for p in range(1, pages + 1)]
    chord(header)(finalize_run.s(run_id))
    return run_id
