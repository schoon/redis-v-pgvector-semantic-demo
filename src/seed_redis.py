"""
Loads the embedded corpus into Redis: two request indexes (FLAT and
HNSW, same underlying data), the procedure and SOP hybrid-search indexes,
and the semantic router. Customer data is generated (see generate.py)
for narrative color but isn't loaded here — no scenario in this demo
queries requests scoped to one customer, so there's nothing for that
file to join against.
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

    print("Flushing and creating indexes (req_flat_idx, procedure_idx, sop_idx)...")
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

    print("Building semantic router (embedding route reference utterances)...")
    t0 = time.time()
    rs.build_router(client, overwrite=True)
    print(f"  router built in {time.time() - t0:.1f}s")

    print("\nWarming (first query after index creation carries a one-time cold-start cost)...")
    rs.vector_search_flat(client, "warm up query", limit=1)
    rs.vector_search_hnsw(client, "warm up query", limit=1)
    rs.semantic_search_procedures(client, "warm up query", limit=1)
    rs.semantic_search_sop(client, "warm up query", limit=1)
    router = rs.build_router(client, overwrite=False)
    rs.route_query(router, client, "warm up query")
    cache = rs.build_semantic_cache(client, overwrite=False)
    cache.clear()
    print("Warm-up complete.")

    info = rs.architecture_info(client)
    print(f"\n{info}")


if __name__ == "__main__":
    main()
