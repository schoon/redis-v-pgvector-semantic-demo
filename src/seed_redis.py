"""
Loads the embedded corpus into Redis: two transaction indexes (FLAT and
HNSW, same underlying data), the product and FAQ hybrid-search indexes,
and the semantic router. Card/customer data is generated (see
generate.py) for narrative color but isn't loaded here — no scenario in
this demo queries transactions scoped to one customer, so there's
nothing for those two files to join against.
"""

import json
import time

import redis_store as rs
from config import DATA_FILES, TRANSACTION_EMBED_SAMPLE
from embeddings import embed_many


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    client = rs.connect()
    client.ping()
    print(f"Connected to Redis.")

    print("Flushing and creating indexes (tx_flat_idx, product_idx, faq_idx)...")
    client.flushall()
    rs.create_indexes(client)

    print(f"Loading a {TRANSACTION_EMBED_SAMPLE:,}-row sample of transactions...")
    all_tx = read_jsonl(DATA_FILES["transactions"])
    sample = all_tx[:TRANSACTION_EMBED_SAMPLE]
    t0 = time.time()
    vecs = embed_many([t["description"] for t in sample], show_progress_bar=True)
    for t, v in zip(sample, vecs):
        t["embedding"] = v
    print(f"  embedded {len(sample):,} transaction descriptions in {time.time() - t0:.1f}s")
    rs.load_transactions(client, sample)
    print(f"  loaded into tx_flat_idx")

    print("Building tx_hnsw_idx (after the bulk load, not before — see redis_store.py)...")
    t0 = time.time()
    rs.create_hnsw_index(client)
    print(f"  built in {time.time() - t0:.1f}s")

    print("Loading card products...")
    products = read_jsonl(DATA_FILES["card_products"])
    vecs = embed_many([p["description"] for p in products])
    for p, v in zip(products, vecs):
        p["embedding"] = v
    rs.load_products(client, products)
    print(f"  {len(products):,} products loaded into product_idx")

    print("Loading FAQ articles...")
    faqs = read_jsonl(DATA_FILES["faq_articles"])
    vecs = embed_many([f["title"] + ". " + f["body"] for f in faqs])
    for f, v in zip(faqs, vecs):
        f["embedding"] = v
    rs.load_faqs(client, faqs)
    print(f"  {len(faqs):,} articles loaded into faq_idx")

    print("Building semantic router (embedding route reference utterances)...")
    t0 = time.time()
    rs.build_router(client, overwrite=True)
    print(f"  router built in {time.time() - t0:.1f}s")

    print("\nWarming (first query after index creation carries a one-time cold-start cost)...")
    rs.vector_search_flat(client, "warm up query", limit=1)
    rs.vector_search_hnsw(client, "warm up query", limit=1)
    rs.semantic_search_products(client, "warm up query", limit=1)
    router = rs.build_router(client, overwrite=False)
    rs.route_query(router, "warm up query")
    print("Warm-up complete.")

    info = rs.architecture_info(client)
    print(f"\n{info}")


if __name__ == "__main__":
    main()
