import re

from psycopg.rows import dict_row

from .db import get_conn
from .embeddings import embed

CANDIDATES = 50  # how many results each search leg contributes before fusion
RRF_K = 60       # standard RRF damping constant

FILTERS = """
    (%(loc)s::text IS NULL OR location ILIKE '%%' || %(loc)s || '%%')
    AND (NOT %(remote)s OR remote)
"""

VECTOR_SQL = f"""
SELECT id, title, company, location, remote, url,
       1 - (embedding <=> %(q)s) AS score
FROM jobs
WHERE embedding IS NOT NULL AND {FILTERS}
ORDER BY embedding <=> %(q)s
LIMIT %(n)s
"""

# Title matches (weight A) count more than description matches (weight B).
KEYWORD_SQL = f"""
WITH q AS (SELECT to_tsquery('english', %(tsq)s) AS tsq)
SELECT j.id, j.title, j.company, j.location, j.remote, j.url
FROM jobs j, q
WHERE j.fts @@ q.tsq AND {FILTERS}
ORDER BY ts_rank_cd(j.fts, q.tsq) DESC
LIMIT %(n)s
"""


def _key(r: dict) -> tuple:
    """Same job posted under several URLs shares this key."""
    return (r["title"].lower(), (r["company"] or "").lower(), (r["location"] or "").lower())


def _tsquery(text: str) -> str:
    words = re.findall(r"\w+", text)[:12]
    return " | ".join(words)  # OR semantics: any word can match


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if _key(r) not in seen:
            seen.add(_key(r))
            out.append(r)
    return out


def _rrf(*ranked_lists: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (RRF_K + rank)."""
    fused: dict[tuple, dict] = {}
    for rows in ranked_lists:
        seen, rank = set(), 0
        for r in rows:
            key = _key(r)
            if key in seen:  # duplicate posting: only its best rank counts
                continue
            seen.add(key)
            rank += 1
            entry = fused.setdefault(key, {**r, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)


def search_jobs(
    query: str,
    location: str | None = None,
    remote_only: bool = False,
    k: int = 5,
    mode: str = "hybrid",  # "hybrid" or "vector"
) -> list[dict]:
    qvec = embed([query])[0]
    params = {"loc": location, "remote": remote_only, "n": CANDIDATES}

    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(VECTOR_SQL, {**params, "q": qvec})
        vector_rows = cur.fetchall()

        keyword_rows: list[dict] = []
        tsq = _tsquery(query)
        if mode == "hybrid" and tsq:
            cur.execute(KEYWORD_SQL, {**params, "tsq": tsq})
            keyword_rows = cur.fetchall()

    if mode == "vector":
        return _dedupe(vector_rows)[:k]

    results = _rrf(vector_rows, keyword_rows)[:k]
    for r in results:
        r["score"] = round(r["score"], 4)
    return results


def get_jobs(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, title, company, location, remote, url FROM jobs WHERE id = ANY(%s)",
            (ids,),
        )
        return cur.fetchall()