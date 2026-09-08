"""
Loads the embedded corpus into Redis: two request indexes (FLAT and
HNSW, same underlying data), the procedure and SOP hybrid-search
indexes, the customer index + TIN Bloom filter + exact TIN index for
duplicate-identity detection, and the semantic router.
"""

import json
import time

import redis_store as rs
from config import DATA_FILES, REQUEST_EMBED_SAMPLE
from embeddings import embed_many


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    client = rs.connect()
    client.ping()
    print(f"Connected to Redis.")

    print("Flushing and creating indexes (req_flat_idx, procedure_idx, sop_idx, customer_idx)...")
    client.flushall()
    rs.create_indexes(client)

    print(f"Loading a {REQUEST_EMBED_SAMPLE:,}-row sample of requests...")
    all_requests = read_jsonl(DATA_FILES["requests"])
    sample = all_requests[:REQUEST_EMBED_SAMPLE]
    t0 = time.time()
    vecs = embed_many([r["description"] for r in sample], show_progress_bar=True)
    for r, v in zip(sample, vecs):
        r["embedding"] = v
    print(f"  embedded {len(sample):,} request descriptions in {time.time() - t0:.1f}s")
    rs.load_requests(client, sample)
    print(f"  loaded into req_flat_idx")

    print("Building req_hnsw_idx (after the bulk load, not before — see redis_store.py)...")
    t0 = time.time()
    rs.create_hnsw_index(client)
    print(f"  built in {time.time() - t0:.1f}s")

    print("Loading procedure catalog...")
    procedures = read_jsonl(DATA_FILES["procedures"])
    vecs = embed_many([p["description"] for p in procedures])
    for p, v in zip(procedures, vecs):
        p["embedding"] = v
    rs.load_procedures(client, procedures)
    print(f"  {len(procedures):,} procedures loaded into procedure_idx")

    print("Loading SOP articles...")
    sops = read_jsonl(DATA_FILES["sop_articles"])
    vecs = embed_many([s["title"] + ". " + s["body"] for s in sops])
    for s, v in zip(sops, vecs):
        s["embedding"] = v
    rs.load_sops(client, sops)
    print(f"  {len(sops):,} articles loaded into sop_idx")

    print("Loading customers (for duplicate detection)...")
    customers = read_jsonl(DATA_FILES["customers"])
    # Embedded on name only -- city/state are filtered on exactly in
    # check_duplicate(), not blended into the fuzzy-matched text. See
    # redis_store.py's customer_idx schema comment for why.
    vecs = embed_many([c["name"] for c in customers])
    for c, v in zip(customers, vecs):
        c["embedding"] = v
    rs.load_customers(client, customers)
    print(f"  {len(customers):,} customers loaded into customer_idx")

    print("Building TIN bloom filter + exact TIN index...")
    rs.build_tin_bloom(client, [c["tin"] for c in customers])
    rs.load_tin_index(client, customers)
    print("  done")

    print("Building semantic router (embedding route reference utterances)...")
    t0 = time.time()
    rs.build_router(client, overwrite=True)
    print(f"  router built in {time.time() - t0:.1f}s")

    print("\nWarming (first query after index creation carries a one-time cold-start cost)...")
    rs.vector_search_flat(client, "warm up query", limit=1)
    rs.vector_search_hnsw(client, "warm up query", limit=1)
    rs.semantic_search_procedures(client, "warm up query", limit=1)
    rs.semantic_search_sop(client, "warm up query", limit=1)
    rs.check_duplicate(client, "Warm Up", "Nowhere", "ZZ", "000-00-0000")
    router = rs.build_router(client, overwrite=False)
    rs.route_query(router, client, "warm up query")
    cache = rs.build_semantic_cache(client, overwrite=False)
    cache.clear()
    print("Warm-up complete.")

    info = rs.architecture_info(client)
    print(f"\n{info}")


if __name__ == "__main__":
    main()
