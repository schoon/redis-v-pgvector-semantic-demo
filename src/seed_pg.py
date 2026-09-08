"""
Loads the same embedded corpus into Postgres: transactions (with an HNSW
index built after the bulk load, never before — building it row-by-row
during insert is far slower), the product/FAQ tables, and the route
reference table for the hand-rolled router. Same sample, same
descriptions, same embeddings as seed_redis.py.
"""

import json
import time

import pg_store as pgs
from config import DATA_FILES, TRANSACTION_EMBED_SAMPLE
from embeddings import embed_many


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    conn = pgs.connect()
    print("Connected to Postgres.")

    print("Creating schema (transactions, card_products, faq_articles, routes)...")
    pgs.create_schema(conn)

    print(f"Loading a {TRANSACTION_EMBED_SAMPLE:,}-row sample of transactions...")
    all_tx = read_jsonl(DATA_FILES["transactions"])
    sample = all_tx[:TRANSACTION_EMBED_SAMPLE]
    t0 = time.time()
    vecs = embed_many([t["description"] for t in sample], show_progress_bar=True)
    for t, v in zip(sample, vecs):
        t["embedding"] = v
    print(f"  embedded {len(sample):,} transaction descriptions in {time.time() - t0:.1f}s")
    pgs.load_transactions(conn, sample)
    print("  loaded into transactions")

    print("Building HNSW index on transactions.embedding...")
    t0 = time.time()
    pgs.create_hnsw_index(conn)
    print(f"  built in {time.time() - t0:.1f}s")

    print("Loading card products...")
    products = read_jsonl(DATA_FILES["card_products"])
    vecs = embed_many([p["description"] for p in products])
    for p, v in zip(products, vecs):
        p["embedding"] = v
    pgs.load_products(conn, products)
    print(f"  {len(products):,} products loaded")

    print("Loading FAQ articles...")
    faqs = read_jsonl(DATA_FILES["faq_articles"])
    vecs = embed_many([f["title"] + ". " + f["body"] for f in faqs])
    for f, v in zip(faqs, vecs):
        f["embedding"] = v
    pgs.load_faqs(conn, faqs)
    print(f"  {len(faqs):,} articles loaded")

    print("Loading routing table (routes + reference utterances)...")
    pgs.load_routes(conn)
    print("  done")

    print("\nRunning ANALYZE...")
    with conn.cursor() as cur:
        cur.execute("ANALYZE transactions, card_products, faq_articles, route_references")

    info = pgs.architecture_info(conn)
    print(f"\n{info}")


if __name__ == "__main__":
    main()
