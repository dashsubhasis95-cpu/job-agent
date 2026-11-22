"""Retrieval eval: compares vector-only search with hybrid (vector + full-text, RRF).

Relevance is judged by keyword match on the job title.
  hit@k        = share of queries with at least one relevant job in the top k
  precision@k  = share of top-k results that are relevant (keyword-judged)
  latency      = end-to-end search time, p50 / p95

Run from the project root:  python -m evals.run_eval
"""
import json
import re
import statistics
import time
from pathlib import Path

from app.search import search_jobs

K = 5
HERE = Path(__file__).parent


def is_relevant(title: str, keywords: list[str]) -> bool:
    """A keyword counts only when it starts a word (so 'it' does not match 'Audit')."""
    t = title.lower()
    return any(re.search(rf"\b{re.escape(kw.strip())}", t) for kw in keywords)


def run(mode: str, queries: list[dict]) -> dict:
    hits, precisions, latencies, per_query = 0, [], [], []
    for item in queries:
        start = time.perf_counter()
        results = search_jobs(item["query"], k=K, mode=mode)
        latencies.append((time.perf_counter() - start) * 1000)

        flags = [is_relevant(r["title"], item["keywords"]) for r in results]
        precision = sum(flags) / max(len(flags), 1)
        hits += any(flags)
        precisions.append(precision)
        per_query.append(
            {"query": item["query"], "precision": round(precision, 2), "titles": [r["title"] for r in results]}
        )

    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    return {
        "queries": len(queries),
        f"hit@{K}": round(hits / len(queries), 3),
        f"precision@{K}": round(statistics.mean(precisions), 3),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(p95, 1),
        "weakest_queries": sorted(per_query, key=lambda x: x["precision"])[:5],
        "_per_query": per_query,
    }


def main():
    queries = json.loads((HERE / "queries.json").read_text())
    search_jobs("warmup", k=1)  # load the embedding model before timing

    results = {mode: run(mode, queries) for mode in ("vector", "hybrid")}

    # queries where hybrid beat or lost to vector-only
    changes = []
    for v, h in zip(results["vector"]["_per_query"], results["hybrid"]["_per_query"]):
        if v["precision"] != h["precision"]:
            changes.append({"query": v["query"], "vector": v["precision"], "hybrid": h["precision"]})
    for r in results.values():
        del r["_per_query"]

    output = {
        "comparison": {
            "vector_precision": results["vector"][f"precision@{K}"],
            "hybrid_precision": results["hybrid"][f"precision@{K}"],
            "queries_changed": sorted(changes, key=lambda c: c["hybrid"] - c["vector"]),
        },
        **results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    (HERE / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()