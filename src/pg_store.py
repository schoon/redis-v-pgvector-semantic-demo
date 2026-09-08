"""
Every Postgres / pgvector operation this demo runs, in one place.

One `embedding vector(384)` column on `transactions`, with a single HNSW
index on it (`CREATE INDEX ... USING hnsw`). That one index covers the
"HNSW" scenario directly. For the "FLAT / exact" scenario there is no
separate index to point at — pgvector's exact equivalent is simply not
using the HNSW index at all, so `vector_search_flat()` disables index
scans for that one query (`SET LOCAL enable_indexscan/enable_bitmapscan
= off`) and falls back to a sequential scan that checks every row. This
is the standard, documented way to force an exact nearest-neighbor
search in Postgres once an approximate index exists on the same column.

Semantic routing has no built-in primitive in Postgres/pgvector — it's
hand-rolled here: reference utterances per route live in
`route_references`, and a query is classified by averaging cosine
distance to each route's references (matching RedisVL's own default
aggregation, so the two engines are doing the identical calculation, not
just a similar one) and taking the route with the lowest average,
subject to that route's distance threshold.
"""

import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from config import PG_DSN
from embeddings import embed
from routes import ROUTES


def _vec(v):
    # register_vector() only wires up dumpers for numpy.ndarray (and
    # pgvector's own Vector wrapper) — a plain Python list of floats, which
    # is what embeddings.embed() returns everywhere else in this repo, gets
    # sent as a generic double precision[] and the <=> operator has no
    # overload for that against a vector column. Cast at the pg_store
    # boundary rather than changing what embed() returns, since RedisVL's
    # CustomTextVectorizer on the other side specifically requires plain
    # float lists.
    return np.asarray(v, dtype="float32")

# HNSW build/query defaults. m/ef_construction control index build
# quality vs. build time; ef_search controls query-time recall vs. speed.
# These are pgvector's own suggested starting points, not tuned for this
# demo specifically.
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 40


def connect():
    conn = psycopg.connect(PG_DSN, autocommit=True)
    # On a brand-new database the `vector` type doesn't exist yet —
    # register_vector() fetches its OID from pg_type and fails outright
    # if the extension was never created, so this has to happen on every
    # connect(), not just once in create_schema(), or the very first
    # connection to a fresh container fails before schema setup even runs.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def create_schema(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("DROP TABLE IF EXISTS transactions CASCADE")
        cur.execute("""
            CREATE TABLE transactions (
                transaction_id TEXT PRIMARY KEY,
                merchant TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                amount NUMERIC NOT NULL,
                txn_date DATE NOT NULL,
                embedding vector(384) NOT NULL
            )
        """)
        cur.execute("CREATE INDEX transactions_category_ix ON transactions (category)")

        cur.execute("DROP TABLE IF EXISTS card_products CASCADE")
        cur.execute("""
            CREATE TABLE card_products (
                product_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                annual_fee NUMERIC NOT NULL,
                description TEXT NOT NULL,
                embedding vector(384) NOT NULL
            )
        """)
        cur.execute("CREATE INDEX card_products_category_ix ON card_products (category)")
        cur.execute("CREATE INDEX card_products_fee_ix ON card_products (annual_fee)")

        cur.execute("DROP TABLE IF EXISTS faq_articles CASCADE")
        cur.execute("""
            CREATE TABLE faq_articles (
                article_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                embedding vector(384) NOT NULL
            )
        """)
        cur.execute("CREATE INDEX faq_articles_category_ix ON faq_articles (category)")

        cur.execute("DROP TABLE IF EXISTS route_references CASCADE")
        cur.execute("DROP TABLE IF EXISTS routes CASCADE")
        cur.execute("""
            CREATE TABLE routes (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                action TEXT NOT NULL,
                distance_threshold REAL NOT NULL,
                search_target TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE route_references (
                id SERIAL PRIMARY KEY,
                route_name TEXT NOT NULL REFERENCES routes (name),
                reference TEXT NOT NULL,
                embedding vector(384) NOT NULL
            )
        """)
        cur.execute("CREATE INDEX route_references_route_ix ON route_references (route_name)")


# CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) — approximate
# nearest-neighbor graph index. Built once, after the bulk load (building
# it row-by-row during insert would be far slower).
def create_hnsw_index(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE INDEX transactions_embedding_hnsw_ix ON transactions
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})
        """)


def load_transactions(conn, rows, batch_size=2000):
    with conn.cursor() as cur:
        batch = []
        for r in rows:
            batch.append((r["transaction_id"], r["merchant"], r["category"], r["description"], r["amount"], r["date"], _vec(r["embedding"])))
            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT INTO transactions (transaction_id, merchant, category, description, amount, txn_date, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    batch,
                )
                batch = []
        if batch:
            cur.executemany(
                "INSERT INTO transactions (transaction_id, merchant, category, description, amount, txn_date, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                batch,
            )


def load_products(conn, rows):
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO card_products (product_id, name, category, annual_fee, description, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(r["product_id"], r["name"], r["category"], r["annual_fee"], r["description"], _vec(r["embedding"])) for r in rows],
        )


def load_faqs(conn, rows):
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO faq_articles (article_id, category, title, body, embedding) VALUES (%s, %s, %s, %s, %s)",
            [(r["article_id"], r["category"], r["title"], r["body"], _vec(r["embedding"])) for r in rows],
        )


def load_routes(conn):
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO routes (name, description, action, distance_threshold, search_target) VALUES (%s, %s, %s, %s, %s)",
            [(r["name"], r["description"], r["action"], r["distance_threshold"], r["search_target"]) for r in ROUTES],
        )
        refs = []
        for r in ROUTES:
            for ref_text in r["references"]:
                refs.append((r["name"], ref_text, _vec(embed(ref_text))))
        cur.executemany(
            "INSERT INTO route_references (route_name, reference, embedding) VALUES (%s, %s, %s)",
            refs,
        )


def _rows_to_dicts(cur, rows):
    cols = [c.name for c in cur.description]
    return [dict(zip(cols, row)) for row in rows]


# `with client.cursor() as cur:` creates a fresh cursor object on every
# call — measured directly, that costs ~25% more than reusing one cursor
# per connection for a small query (0.24ms vs 0.18ms for a 15-row
# lookup). Every hot-path query below reuses a single cursor cached on
# the connection object instead. This has nothing to do with the
# Redis/Postgres comparison — it's a plain Postgres-side inefficiency,
# and it makes Postgres *faster*, not closer to Redis. Fixed anyway:
# leaving a competitor's inefficiency in place to look better by
# comparison isn't the kind of "fair fight" this demo is for (see
# CLAUDE.md's rule about that).
def _cursor(conn):
    cur = getattr(conn, "_demo_cursor", None)
    if cur is None or cur.closed:
        cur = conn.cursor()
        conn._demo_cursor = cur
    return cur


# Exact nearest-neighbor: sequential scan over every row, computing cosine
# distance for each. Disabling index scans for this one query is the
# documented way to force pgvector to skip the HNSW index and check
# every candidate, matching RedisVL's FLAT algorithm's guarantee.
def vector_search_flat(client, query_text=None, limit=10, vector=None):
    vec = vector if vector is not None else _vec(embed(query_text))
    sql = """
        SELECT transaction_id, merchant, category, description, amount, txn_date,
               1 - (embedding <=> %s) AS score
        FROM transactions
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    cur = _cursor(client)
    with client.transaction():
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        t0 = time.perf_counter()
        cur.execute(sql, (vec, vec, limit))
        rows = _rows_to_dicts(cur, cur.fetchall())
        ms = (time.perf_counter() - t0) * 1000
    return rows, ms, sql.strip()


# Same query, but with the HNSW index available and enable_indexscan left
# alone — the planner uses the index's approximate ordering by embedding
# <=> query, i.e. the same shape of query as the FLAT scenario above, with
# only the access path (and, correspondingly, exactness) different.
def vector_search_hnsw(client, query_text=None, limit=10, vector=None):
    vec = vector if vector is not None else _vec(embed(query_text))
    sql = """
        SELECT transaction_id, merchant, category, description, amount, txn_date,
               1 - (embedding <=> %s) AS score
        FROM transactions
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    cur = _cursor(client)
    cur.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    t0 = time.perf_counter()
    cur.execute(sql, (vec, vec, limit))
    rows = _rows_to_dicts(cur, cur.fetchall())
    ms = (time.perf_counter() - t0) * 1000
    return rows, ms, sql.strip()


def semantic_search_products(client, query_text=None, category=None, max_annual_fee=None, limit=10, vector=None):
    vec = vector if vector is not None else _vec(embed(query_text))
    conds = []
    cond_params = []
    if category:
        conds.append("category = %s")
        cond_params.append(category)
    if max_annual_fee is not None:
        conds.append("annual_fee <= %s")
        cond_params.append(max_annual_fee)
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    sql = f"""
        SELECT product_id, name, category, annual_fee,
               description, 1 - (embedding <=> %s) AS score
        FROM card_products
        {where}
        ORDER BY embedding <=> %s
        LIMIT %s
    """.strip()
    params = [vec, *cond_params, vec, limit]
    cur = _cursor(client)
    t0 = time.perf_counter()
    cur.execute(sql, params)
    rows = _rows_to_dicts(cur, cur.fetchall())
    ms = (time.perf_counter() - t0) * 1000
    return rows, ms, sql


def semantic_search_faq(client, query_text=None, category=None, limit=10, vector=None):
    vec = vector if vector is not None else _vec(embed(query_text))
    where = "WHERE category = %s" if category else ""
    params = [vec] + ([category] if category else []) + [vec, limit]
    sql = f"""
        SELECT article_id, category, title, body, 1 - (embedding <=> %s) AS score
        FROM faq_articles
        {where}
        ORDER BY embedding <=> %s
        LIMIT %s
    """.strip()
    cur = _cursor(client)
    t0 = time.perf_counter()
    cur.execute(sql, params)
    rows = _rows_to_dicts(cur, cur.fetchall())
    ms = (time.perf_counter() - t0) * 1000
    return rows, ms, sql


# Hand-rolled semantic routing, replicating RedisVL SemanticRouter's exact
# algorithm rather than a simpler approximation of it — this is a two-stage
# filter, not a plain average:
#   1. Every reference farther than the largest distance_threshold across
#      all routes is dropped BEFORE averaging (RedisVL's RangeQuery does
#      this filtering at the reference level, not the route level).
#   2. Remaining references are averaged per route (aggregation_method
#      "avg", RedisVL's default); a route with zero surviving references
#      can't win at all, regardless of how close its other references are.
#   3. The winning route (lowest average) is only accepted if that average
#      is still under *that route's own* distance_threshold.
# Skipping stage 1 and just averaging every reference (an earlier version
# of this function did that) is a materially different, more lenient
# calculation — it can pick a different winner than RedisVL would for the
# same sentence. No purpose-built primitive for any of this in Postgres —
# every step here is plain SQL and application code.
def route_query(client, query_text):
    vec = _vec(embed(query_text))
    sql = """
        WITH max_threshold AS (
            SELECT MAX(distance_threshold) AS t FROM routes
        ),
        distances AS (
            SELECT rr.route_name, rr.embedding <=> %s AS dist
            FROM route_references rr, max_threshold
            WHERE rr.embedding <=> %s <= max_threshold.t
        )
        SELECT r.name, r.description, r.action, r.distance_threshold, r.search_target,
               AVG(d.dist) AS avg_distance
        FROM distances d
        JOIN routes r ON r.name = d.route_name
        GROUP BY r.name, r.description, r.action, r.distance_threshold, r.search_target
        ORDER BY avg_distance ASC
        LIMIT 1
    """
    cur = _cursor(client)
    t0 = time.perf_counter()
    cur.execute(sql, (vec, vec))
    row = cur.fetchone()
    ms = (time.perf_counter() - t0) * 1000
    no_match = {"route": None, "distance": None, "description": None, "action": None,
                "search_target": None, "search_result": None, "search_ms": None}
    if row is None:
        return no_match, ms
    name, description, action, threshold, search_target, avg_distance = row
    if avg_distance >= threshold:  # RedisVL's own filter is strict "<", matched here
        return {**no_match, "distance": float(avg_distance)}, ms

    # Same "route to the right vector search" step as redis_store.py,
    # reusing the same query embedding — no second embed() call.
    search_result, search_ms = None, None
    if search_target == "products":
        search_result, search_ms, _ = semantic_search_products(client, limit=3, vector=vec)
    elif search_target == "faq":
        search_result, search_ms, _ = semantic_search_faq(client, limit=3, vector=vec)

    return {
        "route": name,
        "distance": float(avg_distance),
        "description": description,
        "action": action,
        "search_target": search_target,
        "search_result": search_result,
        "search_ms": search_ms,
    }, ms


def architecture_info(client):
    info = {}
    with client.cursor() as cur:
        for table in ("transactions", "card_products", "faq_articles", "route_references"):
            cur.execute(f"SELECT count(*) FROM {table}")
            info[table] = {"num_rows": cur.fetchone()[0]}
        cur.execute("""
            SELECT indexname, indexdef FROM pg_indexes
            WHERE tablename IN ('transactions', 'card_products', 'faq_articles', 'route_references')
            ORDER BY tablename, indexname
        """)
        info["indexes"] = [{"name": r[0], "def": r[1]} for r in cur.fetchall()]
    return info
