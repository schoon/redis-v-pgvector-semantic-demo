"""
Concurrent vector-search throughput, Redis vs Postgres, engines run
sequentially (never simultaneously). Mirrors the benchmark shape used in
the sibling Node.js demos in this series: fixed concurrency, a warm-up
window, then a measured window, percentiles computed from every request's
latency, not just the mean.

The query vector is embedded ONCE before the timed run and reused by
every worker thread — see redis_store.vector_search_flat/hnsw and
pg_store's equivalents, which accept a precomputed `vector=` to skip
embed() entirely. Embedding is a single-threaded, GIL-bound CPU
operation; if every request re-embedded the same text, concurrent
"throughput" would mostly measure how fast one CPU core can run the
embedding model, not either datastore.

    python src/bench.py
    python src/bench.py --concurrency=16 --duration=8
"""

import argparse
import json
import os
import threading
import time

import numpy as np

import pg_store as pgs
import redis_store as rs
from config import DATA_DIR
from embeddings import embed

QUERY_TEXT = "coffee shop purchases"
RESULTS_FILE = os.path.join(DATA_DIR, "bench-results.json")


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def run_workers(worker_fn, concurrency, warmup_s, duration_s):
    stop_warmup = threading.Event()
    stop_measure = threading.Event()
    latencies_per_thread = [[] for _ in range(concurrency)]
    errors = [0] * concurrency

    def loop(idx, phase_stop, collect):
        client = worker_fn()
        try:
            while not phase_stop.is_set():
                t0 = time.perf_counter()
                try:
                    client()
                except Exception:
                    errors[idx] += 1
                    continue
                ms = (time.perf_counter() - t0) * 1000
                if collect:
                    latencies_per_thread[idx].append(ms)
        finally:
            pass

    # Warm-up window: same load shape, results discarded.
    threads = [threading.Thread(target=loop, args=(i, stop_warmup, False)) for i in range(concurrency)]
    for t in threads:
        t.start()
    time.sleep(warmup_s)
    stop_warmup.set()
    for t in threads:
        t.join()

    # Measured window.
    threads = [threading.Thread(target=loop, args=(i, stop_measure, True)) for i in range(concurrency)]
    wall_t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop_measure.set()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - wall_t0

    all_latencies = sorted(l for thread_lats in latencies_per_thread for l in thread_lats)
    total_errors = sum(errors)
    n = len(all_latencies)
    return {
        "qps": round(n / wall_s),
        "p50": round(percentile(all_latencies, 0.50), 2),
        "p95": round(percentile(all_latencies, 0.95), 2),
        "p99": round(percentile(all_latencies, 0.99), 2),
        "p999": round(percentile(all_latencies, 0.999), 2),
        "max": round(all_latencies[-1], 2) if all_latencies else 0,
        "errors": total_errors,
    }


def bench_redis(vec, concurrency, warmup_s, duration_s):
    def make_worker():
        client = rs.connect()
        return lambda: rs.vector_search_flat(client, limit=8, vector=vec)
    return run_workers(make_worker, concurrency, warmup_s, duration_s)


def bench_postgres(vec, concurrency, warmup_s, duration_s):
    pgvec = np.asarray(vec, dtype="float32")

    def make_worker():
        conn = pgs.connect()
        return lambda: pgs.vector_search_flat(conn, limit=8, vector=pgvec)
    return run_workers(make_worker, concurrency, warmup_s, duration_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    print("Concurrent throughput — exact (FLAT) vector search over 40,000 transactions")
    print(f"  concurrency: {args.concurrency} threads per engine")
    print(f"  duration:    {args.duration}s measured, {args.warmup}s warm-up")
    print("  engines run sequentially, never simultaneously\n")

    vec = embed(QUERY_TEXT)

    print("  redis: warming up... measuring...")
    redis_result = bench_redis(vec, args.concurrency, args.warmup, args.duration)
    print("  redis: done")

    print("  postgres: warming up... measuring...")
    postgres_result = bench_postgres(vec, args.concurrency, args.warmup, args.duration)
    print("  postgres: done\n")

    header = f"{'':10}{'QPS':>10}{'p50':>10}{'p95':>10}{'p99':>10}{'p99.9':>10}{'max':>10}{'errors':>10}"
    print(header)
    for name, r in (("redis", redis_result), ("postgres", postgres_result)):
        print(f"{name:10}{r['qps']:>10}{r['p50']:>10}{r['p95']:>10}{r['p99']:>10}{r['p999']:>10}{r['max']:>10}{r['errors']:>10}")

    ratio = redis_result["qps"] / postgres_result["qps"] if postgres_result["qps"] else float("inf")
    print(f"\n  throughput ratio: {ratio:.2f}x (redis / postgres)")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "concurrency": args.concurrency,
            "durationSec": args.duration,
            "query": QUERY_TEXT,
            "engines": {"redis": redis_result, "postgres": postgres_result},
        }, f, indent=2)
    print(f"\n  results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
