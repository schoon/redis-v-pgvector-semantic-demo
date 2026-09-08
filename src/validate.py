"""
Independent validation. The point, as with the sibling demos in this
series, is NOT "do the two engines agree" — they could agree and both be
wrong, especially here where both are seeded from the exact same
embeddings module. Every expected answer below is recomputed from
scratch in this file (brute-force cosine distance in plain Python/numpy,
or a from-scratch routing implementation), independent of both
redis_store.py and pg_store.py, then compared against what each engine
actually returns.

    python src/validate.py
"""

import json

import numpy as np

import pg_store as pgs
import redis_store as rs
from config import DATA_FILES, REQUEST_EMBED_SAMPLE
from embeddings import embed, embed_many
from routes import ROUTES

results = []


def record(check, passed, detail):
    results.append((check, passed))
    tag = " ok " if passed else "FAIL"
    print(f"  [{tag}] {check.ljust(52)} {detail}")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def cosine_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def brute_force_top_k(query_vec, corpus_vecs, corpus_ids, k):
    q = np.asarray(query_vec)
    q = q / np.linalg.norm(q)
    mat = np.asarray(corpus_vecs)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    distances = 1 - mat.dot(q)
    order = np.argsort(distances)[:k]
    return [corpus_ids[i] for i in order]


def ground_truth_distances(query_vec, corpus_vecs, corpus_ids):
    q = np.asarray(query_vec)
    q = q / np.linalg.norm(q)
    mat = np.asarray(corpus_vecs)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    distances = 1 - mat.dot(q)
    return dict(zip(corpus_ids, distances))


# Many requests share literally identical description text (and so
# identical embeddings) — "top 8 by exact ID" is ambiguous whenever more
# than 8 rows tie at the k-th smallest distance, and a naive set-equality
# check flags that ambiguity as a mismatch even though every ID returned
# is a perfectly correct answer. The right check is: does every ID the
# engine returned have a ground-truth distance at or below the k-th
# smallest ground-truth distance (i.e. is it a legitimate top-k-or-tied
# candidate), not "is it the exact same ID set numpy's argsort happened
# to pick."
def is_valid_top_k(returned_ids, truth_distances, k, eps=1e-4):
    if not returned_ids:
        return False
    kth = sorted(truth_distances.values())[min(k, len(truth_distances)) - 1]
    return all(truth_distances[i] <= kth + eps for i in returned_ids)


# Independent semantic router, written from scratch (not imported from
# pg_store.py or redis_store.py): average distance to a route's
# references, restricted to references within the largest threshold
# across all routes, lowest surviving average wins its own threshold.
def independent_route(query_vec, route_refs_by_name):
    global_threshold = max(r["distance_threshold"] for r in ROUTES)
    best_name, best_avg = None, None
    for r in ROUTES:
        dists = [cosine_distance(query_vec, v) for v in route_refs_by_name[r["name"]]]
        kept = [d for d in dists if d <= global_threshold]
        if not kept:
            continue
        avg = sum(kept) / len(kept)
        if best_avg is None or avg < best_avg:
            best_name, best_avg = r["name"], avg
    if best_name is None:
        return None, None
    threshold = next(r["distance_threshold"] for r in ROUTES if r["name"] == best_name)
    if best_avg >= threshold:
        return None, best_avg
    return best_name, best_avg


def main():
    redis_client = rs.connect()
    pg_conn = pgs.connect()

    print("Checking corpus counts...")
    all_requests = read_jsonl(DATA_FILES["requests"])
    sample_requests = all_requests[:REQUEST_EMBED_SAMPLE]
    procedures = read_jsonl(DATA_FILES["procedures"])
    sops = read_jsonl(DATA_FILES["sop_articles"])

    redis_info = rs.architecture_info(redis_client)
    pg_info = pgs.architecture_info(pg_conn)
    record("Redis req_flat_idx doc count == embedded sample", redis_info["req_flat_idx"]["num_docs"] == len(sample_requests),
           f"{redis_info['req_flat_idx']['num_docs']:,} vs {len(sample_requests):,}")
    record("Postgres requests row count == embedded sample", pg_info["requests"]["num_rows"] == len(sample_requests),
           f"{pg_info['requests']['num_rows']:,} vs {len(sample_requests):,}")
    record("Redis procedure_idx doc count == corpus file", redis_info["procedure_idx"]["num_docs"] == len(procedures),
           f"{redis_info['procedure_idx']['num_docs']} vs {len(procedures)}")
    record("Redis sop_idx doc count == corpus file", redis_info["sop_idx"]["num_docs"] == len(sops),
           f"{redis_info['sop_idx']['num_docs']} vs {len(sops)}")

    # ---- FLAT vector search vs brute-force ground truth -------------------
    print("\nRe-embedding the request sample independently for brute-force ground truth (this takes a few seconds)...")
    req_vecs = embed_many([r["description"] for r in sample_requests])
    req_ids = [r["request_id"] for r in sample_requests]

    queries = ["address change requests", "duplicate TIN cases", "beneficial ownership updates", "due diligence refresh", "deceased customer processing"]
    flat_checked, flat_mismatch = 0, 0
    for q in queries:
        qvec = embed(q)
        truth_dists = ground_truth_distances(qvec, req_vecs, req_ids)
        r_rows, _, _ = rs.vector_search_flat(redis_client, q, limit=8)
        p_rows, _, _ = pgs.vector_search_flat(pg_conn, q, limit=8)
        r_ids = [row["request_id"] for row in r_rows]
        p_ids = [row["request_id"] for row in p_rows]
        flat_checked += 1
        if not is_valid_top_k(r_ids, truth_dists, 8) or not is_valid_top_k(p_ids, truth_dists, 8):
            flat_mismatch += 1
    record("FLAT search == brute-force ground truth (both engines, tie-aware)", flat_mismatch == 0,
           f"{flat_checked} queries checked, {flat_mismatch} mismatches")

    # ---- Procedure semantic search vs brute-force --------------------------
    proc_vecs = embed_many([p["description"] for p in procedures])
    proc_ids = [p["procedure_id"] for p in procedures]
    proc_queries = ["process for a business address change", "how to handle a deceased customer's account", "steps for a duplicate TIN merge"]
    proc_checked, proc_mismatch = 0, 0
    for q in proc_queries:
        qvec = embed(q)
        truth = brute_force_top_k(qvec, proc_vecs, proc_ids, 3)
        r_rows, _, _ = rs.semantic_search_procedures(redis_client, q, limit=3)
        p_rows, _, _ = pgs.semantic_search_procedures(pg_conn, q, limit=3)
        r_ids = [row["procedure_id"] for row in r_rows]
        p_ids = [row["procedure_id"] for row in p_rows]
        proc_checked += 1
        if set(r_ids) != set(truth) or set(p_ids) != set(truth):
            proc_mismatch += 1
    record("Procedure semantic search == brute-force ground truth", proc_mismatch == 0,
           f"{proc_checked} queries checked, {proc_mismatch} mismatches")

    # ---- Semantic routing vs an independently-written router --------------
    print("\nRe-embedding route references independently for ground-truth routing...")
    route_refs_by_name = {r["name"]: embed_many(r["references"]) for r in ROUTES}
    router = rs.build_router(redis_client, overwrite=False)

    test_utterances = []
    for r in ROUTES[:6]:
        test_utterances.append(r["references"][0])  # a route's own reference: must self-classify
    test_utterances += [
        "I want someone to look into a case that's been open forever",
        "asdkjqwoe random gibberish text with no clear intent",
    ]

    route_checked, route_mismatch, agree_checked, agree_mismatch = 0, 0, 0, 0
    search_checked, search_mismatch = 0, 0
    route_key = lambda row: row.get("procedure_id") or row.get("sop_id")
    for utt in test_utterances:
        qvec = embed(utt)
        truth_name, _ = independent_route(qvec, route_refs_by_name)
        r_match, _ = rs.route_query(router, redis_client, utt)
        p_match, _ = pgs.route_query(pg_conn, utt)
        route_checked += 1
        if r_match["route"] != truth_name or p_match["route"] != truth_name:
            route_mismatch += 1
        agree_checked += 1
        if r_match["route"] != p_match["route"]:
            agree_mismatch += 1

        target = r_match.get("search_target")
        if target in ("procedures", "sop"):
            search_checked += 1
            r_top = route_key(r_match["search_result"][0]) if r_match["search_result"] else None
            p_top = route_key(p_match["search_result"][0]) if p_match["search_result"] else None
            if r_top is None or r_top != p_top:
                search_mismatch += 1
    record("Routing == independently-written ground truth (both engines)", route_mismatch == 0,
           f"{route_checked} utterances checked, {route_mismatch} mismatches")
    record("Redis and Postgres agree on every routing decision", agree_mismatch == 0,
           f"{agree_checked} utterances, {agree_mismatch} disagreements")
    record("Route-to-search: both engines' top hit agrees", search_mismatch == 0,
           f"{search_checked} routed utterances checked, {search_mismatch} mismatches")

    failed = [c for c, p in results if not p]
    print(f"\n  {len(results)} checks · {len(results) - len(failed)} passed · {len(failed)} failed")
    if failed:
        print("\n  FAILURES:")
        for c in failed:
            print(f"    {c}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
