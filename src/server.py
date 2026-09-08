"""
FastAPI app exposing one endpoint per comparison scenario. Every search
endpoint runs both engines for the identical query and returns
{redis: {ms, rows, query}, postgres: {ms, rows, query}, ...} — never a
precomputed "winner" field; the frontend computes who won from the raw
ms values it's already displaying; see public/index.html's setVerdict(),
which exists specifically so this demo never repeats the mistake (made
and fixed in a sibling demo) of asserting a winner that the visible
numbers on screen contradict.
"""

import json
import os
import statistics

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import pg_store as pgs
import redis_store as rs
from config import PORT

app = FastAPI()

redis_client = rs.connect()
pg_conn = pgs.connect()
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
def vector_search(query: str = "coffee purchase", algorithm: str = "flat", limit: int = 10, runs: str = "3"):
    n = clamp_runs(runs)
    if algorithm == "hnsw":
        r_rows, r_ms, r_q = median_timed(lambda: rs.vector_search_hnsw(redis_client, query, limit), n)
        p_rows, p_ms, p_q = median_timed(lambda: pgs.vector_search_hnsw(pg_conn, query, limit), n)
        # Recall context: how much does this approximate result overlap
        # with that SAME engine's exact result for the identical query?
        r_flat_rows, _, _ = rs.vector_search_flat(redis_client, query, limit)
        p_flat_rows, _, _ = pgs.vector_search_flat(pg_conn, query, limit)
        r_recall = _recall(r_rows, r_flat_rows)
        p_recall = _recall(p_rows, p_flat_rows)
    else:
        r_rows, r_ms, r_q = median_timed(lambda: rs.vector_search_flat(redis_client, query, limit), n)
        p_rows, p_ms, p_q = median_timed(lambda: pgs.vector_search_flat(pg_conn, query, limit), n)
        r_recall = p_recall = None

    return {
        "query": query, "algorithm": algorithm, "runs": n,
        "redis": {"ms": r_ms, "rows": r_rows, "query": r_q, "recall_vs_exact": r_recall},
        "postgres": {"ms": p_ms, "rows": p_rows, "query": p_q, "recall_vs_exact": p_recall},
        "countsMatch": len(r_rows) == len(p_rows),
    }


def _recall(approx_rows, exact_rows):
    exact_ids = {r["transaction_id"] for r in exact_rows}
    if not exact_ids:
        return None
    hit = sum(1 for r in approx_rows if r["transaction_id"] in exact_ids)
    return round(hit / len(exact_ids), 3)


@app.get("/api/semantic-search")
def semantic_search(query: str = "no annual fee travel card", corpus: str = "products",
                     category: str = "", max_annual_fee: str = "", limit: int = 10, runs: str = "3"):
    n = clamp_runs(runs)
    max_fee = float(max_annual_fee) if max_annual_fee else None
    cat = category or None

    if corpus == "faq":
        r_rows, r_ms, r_q = median_timed(lambda: rs.semantic_search_faq(redis_client, query, cat, limit), n)
        p_rows, p_ms, p_q = median_timed(lambda: pgs.semantic_search_faq(pg_conn, query, cat, limit), n)
    else:
        r_rows, r_ms, r_q = median_timed(lambda: rs.semantic_search_products(redis_client, query, cat, max_fee, limit), n)
        p_rows, p_ms, p_q = median_timed(lambda: pgs.semantic_search_products(pg_conn, query, cat, max_fee, limit), n)

    return {
        "query": query, "corpus": corpus, "category": cat, "max_annual_fee": max_fee, "runs": n,
        "redis": {"ms": r_ms, "rows": r_rows, "query": r_q},
        "postgres": {"ms": p_ms, "rows": p_rows, "query": p_q},
        "countsMatch": len(r_rows) == len(p_rows),
    }


@app.get("/api/semantic-route")
def semantic_route(query: str = "I think someone stole my card", runs: str = "3"):
    n = clamp_runs(runs)

    def r_once():
        match, ms = rs.route_query(router, redis_client, query)
        return match, ms, None

    def p_once():
        match, ms = pgs.route_query(pg_conn, query)
        return match, ms, None

    r_match, r_ms, _ = median_timed(r_once, n)
    p_match, p_ms, _ = median_timed(p_once, n)

    return {
        "query": query, "runs": n,
        "redis": {"ms": r_ms, "match": r_match},
        "postgres": {"ms": p_ms, "match": p_match},
        "agree": r_match.get("route") == p_match.get("route"),
    }


@app.get("/api/semantic-cache")
def semantic_cache_demo(query: str = "coffee shop purchases", limit: int = 8, repeats: str = "5"):
    """
    Simulates repeat/popular traffic for the same question: `repeats`
    sequential calls, same query text each time. Redis checks its
    SemanticCache first (call 0 misses and populates it; every call
    after that hits, skipping tx_flat_idx entirely). Postgres has no
    cache, so every single call re-runs the full FLAT scan — there is no
    "first call" vs "repeat call" distinction on that side, on purpose.

    The cache is cleared at the start of every call to this endpoint so
    "call 0" is a genuine miss each time the demo is run, not a hit left
    over from someone else's earlier request.
    """
    n = max(1, min(int(repeats) if str(repeats).isdigit() else 5, 20))
    semantic_cache.clear()

    calls = []
    for i in range(n):
        r_rows, r_ms, r_hit, r_desc = rs.cached_transaction_search(semantic_cache, redis_client, query, limit=limit)
        p_rows, p_ms, p_query = pgs.vector_search_flat(pg_conn, query, limit=limit)
        calls.append({"i": i, "redis_ms": r_ms, "redis_hit": r_hit, "redis_desc": r_desc, "postgres_ms": p_ms})

    repeat_calls = calls[1:] if n > 1 else calls
    return {
        "query": query, "repeats": n, "calls": calls,
        "redis": {
            "rows": r_rows, "query": r_desc,
            "first_ms": calls[0]["redis_ms"],
            "repeat_avg_ms": statistics.mean(c["redis_ms"] for c in repeat_calls),
        },
        "postgres": {
            "rows": p_rows, "query": p_query,
            "first_ms": calls[0]["postgres_ms"],
            "repeat_avg_ms": statistics.mean(c["postgres_ms"] for c in repeat_calls),
        },
        "countsMatch": len(r_rows) == len(p_rows),
    }


@app.get("/api/architecture")
def architecture():
    return {"redis": rs.architecture_info(redis_client), "postgres": pgs.architecture_info(pg_conn)}


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
