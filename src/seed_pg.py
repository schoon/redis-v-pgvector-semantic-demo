"""
Loads the same embedded corpus into Postgres: requests (with an HNSW
index built after the bulk load, never before — building it row-by-row
during insert is far slower), the procedure/SOP tables, and the route
reference table for the hand-rolled router. Same sample, same
descriptions, same embeddings as seed_redis.py.
"""

import json
import time

import pg_store as pgs
from config import DATA_FILES, REQUEST_EMBED_SAMPLE
from embeddings import embed_many


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    conn = pgs.connect()
    print("Connected to Postgres.")

    print("Creating schema (requests, procedures, sop_articles, routes)...")
    pgs.create_schema(conn)

    print(f"Loading a {REQUEST_EMBED_SAMPLE:,}-row sample of requests...")
    all_requests = read_jsonl(DATA_FILES["requests"])
    sample = all_requests[:REQUEST_EMBED_SAMPLE]
    t0 = time.time()
    vecs = embed_many([r["description"] for r in sample], show_progress_bar=True)
    for r, v in zip(sample, vecs):
        r["embedding"] = v
    print(f"  embedded {len(sample):,} request descriptions in {time.time() - t0:.1f}s")
    pgs.load_requests(conn, sample)
    print("  loaded into requests")

    print("Building HNSW index on requests.embedding...")
    t0 = time.time()
    pgs.create_hnsw_index(conn)
    print(f"  built in {time.time() - t0:.1f}s")

    print("Loading procedure catalog...")
    procedures = read_jsonl(DATA_FILES["procedures"])
    vecs = embed_many([p["description"] for p in procedures])
    for p, v in zip(procedures, vecs):
        p["embedding"] = v
    pgs.load_procedures(conn, procedures)
    print(f"  {len(procedures):,} procedures loaded")

    print("Loading SOP articles...")
    sops = read_jsonl(DATA_FILES["sop_articles"])
    vecs = embed_many([s["title"] + ". " + s["body"] for s in sops])
    for s, v in zip(sops, vecs):
        s["embedding"] = v
    pgs.load_sops(conn, sops)
    print(f"  {len(sops):,} articles loaded")

    print("Loading routing table (routes + reference utterances)...")
    pgs.load_routes(conn)
    print("  done")

    print("\nRunning ANALYZE...")
    with conn.cursor() as cur:
        cur.execute("ANALYZE requests, procedures, sop_articles, route_references")

    info = pgs.architecture_info(conn)
    print(f"\n{info}")


if __name__ == "__main__":
    main()
