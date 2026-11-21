import hashlib
import re
from datetime import datetime, timezone

import httpx

from .db import get_conn
from .embeddings import embed

SOURCE = "arbeitnow"
API_URL = "https://www.arbeitnow.com/api/job-board-api"

UPSERT_SQL = """
INSERT INTO jobs (source, external_id, title, company, location, remote,
                  description, url, posted_at, content_hash, embedding, run_id)
VALUES (%(source)s, %(external_id)s, %(title)s, %(company)s, %(location)s, %(remote)s,
        %(description)s, %(url)s, %(posted_at)s, %(hash)s, %(embedding)s, %(run_id)s)
ON CONFLICT (source, external_id) DO UPDATE SET
    title = EXCLUDED.title,
    company = EXCLUDED.company,
    location = EXCLUDED.location,
    remote = EXCLUDED.remote,
    description = EXCLUDED.description,
    url = EXCLUDED.url,
    posted_at = EXCLUDED.posted_at,
    content_hash = EXCLUDED.content_hash,
    embedding = EXCLUDED.embedding,
    run_id = EXCLUDED.run_id,
    updated_at = now()
"""


def clean_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", text).strip()


def fetch_page(page: int) -> list[dict]:
    resp = httpx.get(API_URL, params={"page": page}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _normalize(raw: dict) -> dict | None:
    title = (raw.get("title") or "").strip()
    slug = raw.get("slug")
    if not title or not slug:
        return None
    description = clean_html(raw.get("description", ""))
    company = raw.get("company_name")
    location = raw.get("location")
    created = raw.get("created_at")
    digest = hashlib.sha256(
        f"{title}|{company}|{location}|{description}".encode()
    ).hexdigest()
    return {
        "source": SOURCE,
        "external_id": slug,
        "title": title,
        "company": company,
        "location": location,
        "remote": bool(raw.get("remote")),
        "description": description,
        "url": raw.get("url"),
        "posted_at": datetime.fromtimestamp(created, tz=timezone.utc) if created else None,
        "hash": digest,
    }


def _embed_text(r: dict) -> str:
    return f"{r['title']} at {r['company']}. {r['location']}. {r['description'][:2000]}"


def upsert_jobs(raw_jobs: list[dict], run_id: int) -> tuple[int, int, int]:
    """Returns (seen, inserted, updated). Unchanged jobs are not re-embedded."""
    rows = [r for r in (_normalize(j) for j in raw_jobs) if r]
    if not rows:
        return 0, 0, 0

    with get_conn() as conn:
        existing = dict(
            conn.execute(
                "SELECT external_id, content_hash FROM jobs "
                "WHERE source = %s AND external_id = ANY(%s)",
                (SOURCE, [r["external_id"] for r in rows]),
            ).fetchall()
        )
        changed = [r for r in rows if existing.get(r["external_id"]) != r["hash"]]
        if changed:
            vectors = embed([_embed_text(r) for r in changed])
            with conn.cursor() as cur:
                for r, vec in zip(changed, vectors):
                    cur.execute(UPSERT_SQL, {**r, "embedding": vec, "run_id": run_id})

    inserted = sum(1 for r in changed if r["external_id"] not in existing)
    return len(rows), inserted, len(changed) - inserted
