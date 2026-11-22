from fastapi import FastAPI
from pydantic import BaseModel

from .agent import chat
from .db import get_conn
from .search import get_jobs, search_jobs
from .worker import start_ingestion

app = FastAPI(title="Job Search Agent")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health():
    with get_conn() as conn:
        n = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    return {"status": "ok", "jobs_indexed": n}


@app.post("/ingest")
def ingest(pages: int = 5):
    run_id = start_ingestion.delay(pages).id
    return {"queued": True, "task_id": run_id}


@app.get("/ingest/runs")
def ingest_runs():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, status, jobs_seen, jobs_inserted, jobs_updated, started_at, finished_at "
            "FROM ingestion_runs ORDER BY id DESC LIMIT 10"
        ).fetchall()
    cols = ["id", "status", "jobs_seen", "jobs_inserted", "jobs_updated", "started_at", "finished_at"]
    return [dict(zip(cols, r)) for r in rows]


@app.get("/search")
def search(q: str, location: str | None = None, remote: bool = False, k: int = 5):
    return search_jobs(q, location, remote, k)


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    reply = chat(req.session_id, req.message)
    return {"message": reply.message, "jobs": get_jobs(reply.job_ids)}
