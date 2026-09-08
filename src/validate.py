"""
Independent validation. The point, as with the sibling demos in this
series, is NOT "does Redis's result look plausible" — every expected
answer below is recomputed from scratch in this file (brute-force cosine
distance in plain Python/NumPy, or a from-scratch routing
implementation), independent of redis_store.py, then compared against
what Redis actually returns. Redis being internally consistent with
itself proves nothing on its own; this checks it against ground truth
computed with no help from RedisVL at all.

    python src/validate.py
"""

import json

import numpy as np

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
# is a perfectly correct answer. The right check is: does every ID Redis
# returned have a ground-truth distance at or below the k-th smallest
# ground-truth distance (i.e. is it a legitimate top-k-or-tied
# candidate), not "is it the exact same ID set numpy's argsort happened
# to pick."
def is_valid_top_k(returned_ids, truth_distances, k, eps=1e-4):
    if not returned_ids:
        return False
    kth = sorted(truth_distances.values())[min(k, len(truth_distances)) - 1]
    return all(truth_distances[i] <= kth + eps for i in returned_ids)


# Independent semantic router, written from scratch (not imported from
# redis_store.py): average distance to a route's references, restricted
# to references within the largest threshold across all routes, lowest
# surviving average wins its own threshold. This is RedisVL
# SemanticRouter's exact algorithm, reimplemented independently so
# agreement isn't just "Redis agrees with itself."
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

    print("Checking corpus counts...")
    all_requests = read_jsonl(DATA_FILES["requests"])
    sample_requests = all_requests[:REQUEST_EMBED_SAMPLE]
    procedures = read_jsonl(DATA_FILES["procedures"])
    sops = read_jsonl(DATA_FILES["sop_articles"])

    redis_info = rs.architecture_info(redis_client)
    record("req_flat_idx doc count == embedded sample", redis_info["req_flat_idx"]["num_docs"] == len(sample_requests),
           f"{redis_info['req_flat_idx']['num_docs']:,} vs {len(sample_requests):,}")
    record("procedure_idx doc count == corpus file", redis_info["procedure_idx"]["num_docs"] == len(procedures),
           f"{redis_info['procedure_idx']['num_docs']} vs {len(procedures)}")
    record("sop_idx doc count == corpus file", redis_info["sop_idx"]["num_docs"] == len(sops),
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
        rows, _, _ = rs.vector_search_flat(redis_client, q, limit=8)
        ids = [row["request_id"] for row in rows]
        flat_checked += 1
        if not is_valid_top_k(ids, truth_dists, 8):
            flat_mismatch += 1
    record("FLAT search == brute-force ground truth (tie-aware)", flat_mismatch == 0,
           f"{flat_checked} queries checked, {flat_mismatch} mismatches")

    # ---- Procedure semantic search vs brute-force --------------------------
    proc_vecs = embed_many([p["description"] for p in procedures])
    proc_ids = [p["procedure_id"] for p in procedures]
    proc_queries = ["process for a business address change", "how to handle a deceased customer's account", "steps for a duplicate TIN merge"]
    proc_checked, proc_mismatch = 0, 0
    for q in proc_queries:
        qvec = embed(q)
        truth = brute_force_top_k(qvec, proc_vecs, proc_ids, 3)
        rows, _, _ = rs.semantic_search_procedures(redis_client, q, limit=3)
        ids = [row["procedure_id"] for row in rows]
        proc_checked += 1
        if set(ids) != set(truth):
            proc_mismatch += 1
    record("Procedure semantic search == brute-force ground truth", proc_mismatch == 0,
           f"{proc_checked} queries checked, {proc_mismatch} mismatches")

    # ---- Semantic routing vs an independently-written router --------------
    print("\nRe-embedding route references independently for ground-truth routing...")
    route_refs_by_name = {r["name"]: embed_many(r["references"]) for r in ROUTES}
    router = rs.build_router(redis_client, overwrite=False)

    # Ground truth for the "route to the right search" step: whichever
    # corpus a route's search_target names, the top hit Redis returns
    # after routing must be a legitimate top-1 candidate against the same
    # brute-force distances used above (procedures) or computed fresh
    # (SOP articles) — not just "some result came back."
    sop_vecs = embed_many([s["title"] + ". " + s["body"] for s in sops])
    sop_ids = [s["sop_id"] for s in sops]
    ground_truth_by_corpus = {
        "procedures": (proc_vecs, proc_ids),
        "sop": (sop_vecs, sop_ids),
    }
    route_key_by_corpus = {"procedures": "procedure_id", "sop": "sop_id"}

    test_utterances = []
    for r in ROUTES[:6]:
        test_utterances.append(r["references"][0])  # a route's own reference: must self-classify
    test_utterances += [
        "I want someone to look into a case that's been open forever",
        "asdkjqwoe random gibberish text with no clear intent",
    ]

    route_checked, route_mismatch = 0, 0
    search_checked, search_mismatch = 0, 0
    for utt in test_utterances:
        qvec = embed(utt)
        truth_name, _ = independent_route(qvec, route_refs_by_name)
        match, _ = rs.route_query(router, redis_client, utt)
        route_checked += 1
        if match["route"] != truth_name:
            route_mismatch += 1

        target = match.get("search_target")
        if target in ground_truth_by_corpus and match.get("search_result"):
            search_checked += 1
            corpus_vecs, corpus_ids = ground_truth_by_corpus[target]
            truth_dists = ground_truth_distances(qvec, corpus_vecs, corpus_ids)
            top_id = match["search_result"][0][route_key_by_corpus[target]]
            if not is_valid_top_k([top_id], truth_dists, 1):
                search_mismatch += 1
    record("Routing == independently-written ground truth", route_mismatch == 0,
           f"{route_checked} utterances checked, {route_mismatch} mismatches")
    record("Route-to-search top hit == brute-force ground truth", search_mismatch == 0,
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
