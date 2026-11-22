from psycopg.rows import dict_row

from .db import get_conn
from .embeddings import embed

SEARCH_SQL = """
SELECT id, title, company, location, remote, url,
       1 - (embedding <=> %(q)s) AS score
FROM jobs
WHERE embedding IS NOT NULL
  AND (%(loc)s::text IS NULL OR location ILIKE '%%' || %(loc)s || '%%')
  AND (NOT %(remote)s OR remote)
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
"""

def search_jobs(query: str, location: str | None = None, remote_only: bool = False, k: int = 5) -> list[dict]:
    qvec = embed([query])[0]
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(SEARCH_SQL, {"q": qvec, "loc": location, "remote": remote_only, "k": k * 4})
        rows = cur.fetchall()

    seen, unique = set(), []
    for r in rows:  # same job posted under several URLs: keep the best-scoring copy
        key = (r["title"].lower(), (r["company"] or "").lower(), (r["location"] or "").lower())
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:k]

def get_jobs(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, title, company, location, remote, url FROM jobs WHERE id = ANY(%s)",
            (ids,),
        )
        return cur.fetchall()
