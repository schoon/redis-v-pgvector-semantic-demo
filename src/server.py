"""
FastAPI app exposing one endpoint per Redis/RedisVL search capability
showcased in this demo — exact vector search, approximate vector search,
hybrid semantic search, semantic caching, and semantic routing, all over
a fictitious bank's customer identity operations data. Every endpoint
returns {ms, rows, query, ...} straight from Redis; the frontend's job is
to show that number and explain which Redis feature produced it, not to
race it against anything.
"""

import json
import os
import statistics

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import redis_store as rs
from config import PORT

app = FastAPI()

redis_client = rs.connect()
router = rs.build_router(redis_client, overwrite=False)
semantic_cache = rs.build_semantic_cache(redis_client, overwrite=False)


def median_timed(fn, runs=3):
    """Run fn() `runs` times, return (last_result, median_ms_across_runs)."""
    times = []
    result = None
    for _ in range(runs):
        result_rows, ms, query = fn()
        times.append(ms)
        result = (result_rows, query)
    return result[0], statistics.median(times), result[1]


def clamp_runs(v):
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = 3
    return max(1, min(n, 9))


@app.get("/api/vector-search")
def vector_search(query: str = "address change request", algorithm: str = "flat", limit: int = 10, runs: str = "3"):
    n = clamp_runs(runs)
    if algorithm == "hnsw":
        rows, ms, q = median_timed(lambda: rs.vector_search_hnsw(redis_client, query, limit), n)
        # Recall context: how much does this approximate result overlap
        # with Redis's OWN exact (FLAT) result for the identical query?
        flat_rows, _, _ = rs.vector_search_flat(redis_client, query, limit)
        recall = _recall(rows, flat_rows)
    else:
        rows, ms, q = median_timed(lambda: rs.vector_search_flat(redis_client, query, limit), n)
        recall = None

    return {
        "query": query, "algorithm": algorithm, "runs": n,
        "redis": {"ms": ms, "rows": rows, "query": q, "recall_vs_exact": recall},
    }


def _recall(approx_rows, exact_rows):
    exact_ids = {r["request_id"] for r in exact_rows}
    if not exact_ids:
        return None
    hit = sum(1 for r in approx_rows if r["request_id"] in exact_ids)
    return round(hit / len(exact_ids), 3)


@app.get("/api/semantic-search")
def semantic_search(query: str = "process for a duplicate TIN case", corpus: str = "procedures",
                     category: str = "", max_sla_days: str = "", limit: int = 10, runs: str = "3"):
    n = clamp_runs(runs)
    max_sla = float(max_sla_days) if max_sla_days else None
    cat = category or None

    if corpus == "sop":
        rows, ms, q = median_timed(lambda: rs.semantic_search_sop(redis_client, query, cat, limit), n)
    else:
        rows, ms, q = median_timed(lambda: rs.semantic_search_procedures(redis_client, query, cat, max_sla, limit), n)

    return {
        "query": query, "corpus": corpus, "category": cat, "max_sla_days": max_sla, "runs": n,
        "redis": {"ms": ms, "rows": rows, "query": q},
    }


@app.get("/api/semantic-route")
def semantic_route(query: str = "I think two of my accounts share the same SSN", runs: str = "3"):
    n = clamp_runs(runs)

    def once():
        match, ms = rs.route_query(router, redis_client, query)
        return match, ms, None

    match, ms, _ = median_timed(once, n)

    return {
        "query": query, "runs": n,
        "redis": {"ms": ms, "match": match},
    }


@app.get("/api/semantic-cache")
def semantic_cache_demo(query: str = "customer moved to a new address", limit: int = 8, repeats: str = "5"):
    """
    Simulates repeat/popular traffic for the same question: `repeats`
    sequential calls, same query text each time. RedisVL's SemanticCache
    is checked first on every call — call 0 misses and populates it,
    every call after that hits and never touches req_flat_idx.

    The cache is cleared at the start of every call to this endpoint so
    "call 0" is a genuine miss each time the demo is run, not a hit left
    over from someone else's earlier request.
    """
    n = max(1, min(int(repeats) if str(repeats).isdigit() else 5, 20))
    semantic_cache.clear()

    calls = []
    for i in range(n):
        rows, ms, hit, desc = rs.cached_request_search(semantic_cache, redis_client, query, limit=limit)
        calls.append({"i": i, "ms": ms, "hit": hit, "desc": desc})

    # With repeats=1 there were no repeat calls at all — reporting the
    # single (miss) call's latency as "repeat average" would silently
    # relabel a cache-miss cost as if it reflected the cache-hit
    # benefit, which is the whole point of this scenario. null it out
    # instead and let the frontend say so explicitly.
    repeat_calls = calls[1:]
    return {
        "query": query, "repeats": n, "calls": calls,
        "redis": {
            "rows": rows, "query": desc,
            "first_ms": calls[0]["ms"],
            "repeat_avg_ms": statistics.mean(c["ms"] for c in repeat_calls) if repeat_calls else None,
        },
    }


@app.get("/api/architecture")
def architecture():
    return {"redis": rs.architecture_info(redis_client)}


@app.get("/api/bench-results")
def bench_results():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "bench-results.json")
    if not os.path.exists(path):
        return {"error": "no results"}
    with open(path) as f:
        return json.load(f)


@app.get("/api/routes")
def list_routes():
    from routes import ROUTES
    return [
        {"name": r["name"], "description": r["description"], "sample": r["references"][0]}
        for r in ROUTES
    ]


_PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public")
app.mount("/", StaticFiles(directory=_PUBLIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
