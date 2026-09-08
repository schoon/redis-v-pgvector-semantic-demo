"""
Every Redis / RedisVL operation this demo runs, in one place.

Two indexes cover the same transaction Hashes (`tx:<id>`), each reading
the same `embedding` field but built with a different vector algorithm —
`tx_flat_idx` (FLAT: exact brute-force KNN, scans every candidate) and
`tx_hnsw_idx` (HNSW: approximate nearest-neighbor graph search). Same
data, same field, two independently-built index structures — that's
enough for RediSearch to answer identical queries two different ways, no
duplicated storage required.

`product_idx` and `faq_idx` are small (15 and 26 documents) hybrid
indexes — a vector field plus TAG/NUMERIC fields — used for the
"ask about your card products" semantic-search scenario with filters.

The semantic router uses RedisVL's own `SemanticRouter` extension
untouched, wrapped around the exact same `embeddings.embed`/`embed_many`
functions the rest of this file uses (via `CustomTextVectorizer`), so
"how RedisVL routes a sentence" is not a separately-tuned code path from
"how RedisVL searches a sentence."

`build_semantic_cache()`/`cached_transaction_search()` use RedisVL's
`SemanticCache` extension the same way — same shared vectorizer, no
separate embedding path. There is no Postgres/pgvector equivalent to
compare it against feature-for-feature: pgvector has no caching
primitive, so the Postgres side of that scenario is just
`pg_store.vector_search_flat()`, called fresh every time, exactly like
the plain exact-search scenario. That asymmetry is the point of this
scenario, not an oversight.
"""

import json
import time

import numpy as np
import redis
from redisvl.extensions.llmcache import SemanticCache
from redisvl.extensions.router import Route, SemanticRouter
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Num, Tag
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import CustomTextVectorizer

from config import REDIS_URL, VECTOR_DIM
from embeddings import embed, embed_many
from routes import ROUTES

TX_FLAT_INDEX = "tx_flat_idx"
TX_HNSW_INDEX = "tx_hnsw_idx"
PRODUCT_INDEX = "product_idx"
CACHE_NAME = "tx_search_cache"
CACHE_DISTANCE_THRESHOLD = 0.15
FAQ_INDEX = "faq_idx"
ROUTER_NAME = "intent_router"


def connect():
    return redis.Redis.from_url(REDIS_URL, decode_responses=False)


# RedisVL's own HNSW defaults are M=16, ef_construction=200 (both fine),
# but ef_runtime=10 — measured directly against this corpus, that's too
# low for good recall: a plain-language query like "coffee purchase" (not
# a copy of any stored description) missed its entire target cluster
# and returned "Online Shopping" transactions instead, 0% overlap with
# the exact result. Raising ef_runtime to 40 (matching pgvector's
# hnsw.ef_search=40 below, for a genuinely comparable configuration on
# both engines, not a thumb on the scale) fixed it. Don't drop this back
# to the library default on the theory that it's unnecessary tuning —
# it's the difference between a working and a broken HNSW scenario here.
HNSW_EF_RUNTIME = 40
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200


def _vector_field(algorithm):
    attrs = {
        "algorithm": algorithm,
        "dims": VECTOR_DIM,
        "distance_metric": "cosine",
        "datatype": "float32",
    }
    if algorithm == "hnsw":
        attrs.update({"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_runtime": HNSW_EF_RUNTIME})
    return {"name": "embedding", "type": "vector", "attrs": attrs}


# FT.CREATE tx_flat_idx ON HASH PREFIX tx: SCHEMA id TAG merchant TAG
# category TAG amount NUMERIC embedding VECTOR FLAT ...
# Exact KNN: every candidate vector is compared, no approximation. Correct
# by construction, cost grows linearly with corpus size.
def _tx_schema(index_name, algorithm):
    return IndexSchema.from_dict({
        "index": {"name": index_name, "prefix": "tx:", "storage_type": "hash"},
        "fields": [
            # NOT named "id" — RedisVL query results always include an "id"
            # key of their own (the full Redis key, e.g. "tx::T00000114"),
            # which would collide with and shadow a same-named schema field.
            {"name": "transaction_id", "type": "tag"},
            {"name": "merchant", "type": "tag"},
            {"name": "category", "type": "tag"},
            {"name": "description", "type": "text"},
            {"name": "amount", "type": "numeric"},
            {"name": "date", "type": "tag"},
            _vector_field(algorithm),
        ],
    })


# FT.CREATE product_idx ON HASH PREFIX product: SCHEMA ... embedding
# VECTOR FLAT ... — small corpus (15 rows), FLAT vs HNSW makes no
# measurable difference at this scale, so this scenario is about hybrid
# filtering, not algorithm choice.
def _product_schema():
    return IndexSchema.from_dict({
        "index": {"name": PRODUCT_INDEX, "prefix": "product:", "storage_type": "hash"},
        "fields": [
            {"name": "product_id", "type": "tag"},
            {"name": "name", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "annual_fee", "type": "numeric"},
            {"name": "description", "type": "text"},
            _vector_field("flat"),
        ],
    })


def _faq_schema():
    return IndexSchema.from_dict({
        "index": {"name": FAQ_INDEX, "prefix": "faq:", "storage_type": "hash"},
        "fields": [
            {"name": "article_id", "type": "tag"},
            {"name": "category", "type": "tag"},
            {"name": "title", "type": "text"},
            {"name": "body", "type": "text"},
            _vector_field("flat"),
        ],
    })


def get_index(client, index_name):
    schema = {
        TX_FLAT_INDEX: _tx_schema(TX_FLAT_INDEX, "flat"),
        TX_HNSW_INDEX: _tx_schema(TX_HNSW_INDEX, "hnsw"),
        PRODUCT_INDEX: _product_schema(),
        FAQ_INDEX: _faq_schema(),
    }[index_name]
    return SearchIndex(schema=schema, redis_client=client)


def create_indexes(client):
    # tx_hnsw_idx is deliberately NOT created here — see create_hnsw_index()
    # below for why.
    for name in (TX_FLAT_INDEX, PRODUCT_INDEX, FAQ_INDEX):
        get_index(client, name).create(overwrite=True, drop=True)


# Create tx_hnsw_idx only after load_transactions() has already written
# every tx: hash. Creating it up front, like the other three indexes, and
# letting RediSearch build the HNSW graph incrementally as each document
# is HSET one at a time (the on-write indexing path) measured
# significantly worse recall than building it here, in one shot, against
# already-loaded data (the same background bulk-scan path RediSearch uses
# when you attach an index to an existing keyspace) — a plain-language
# query with no exact match in the corpus ("coffee purchase") missed its
# entire target cluster with the former and found it cleanly with the
# latter, same data, same M/ef_construction/ef_runtime either way. This
# mirrors pgvector's own documented advice to build the HNSW index after
# the bulk load, not before — don't move this back to create_indexes().
def create_hnsw_index(client):
    get_index(client, TX_HNSW_INDEX).create(overwrite=True, drop=True)
    wait_for_indexing(client, TX_HNSW_INDEX)


# The background scan create_hnsw_index() triggers is async — FT.INFO's
# percent_indexed is how you know it's actually done. Querying before it
# reaches 1.0 hits a partially-built graph, which is its own way to get
# degraded recall that has nothing to do with ef_runtime.
def wait_for_indexing(client, index_name, timeout_s=60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = client.ft(index_name).info()
        pct = float(info.get("percent_indexed", 0))
        if pct >= 1.0:
            return
        time.sleep(0.2)
    raise TimeoutError(f"{index_name} did not finish indexing within {timeout_s}s")


def vector_to_bytes(vec):
    return np.array(vec, dtype="float32").tobytes()


def load_transactions(client, rows):
    """rows: iterable of dicts with an added 'embedding' key (list[float])."""
    idx = get_index(client, TX_FLAT_INDEX)  # any of the two works, they share the prefix
    data = [
        {
            "transaction_id": r["transaction_id"],
            "merchant": r["merchant"],
            "category": r["category"],
            "description": r["description"],
            "amount": r["amount"],
            "date": r["date"],
            "embedding": vector_to_bytes(r["embedding"]),
        }
        for r in rows
    ]
    idx.load(data, id_field="transaction_id")


def load_products(client, rows):
    idx = get_index(client, PRODUCT_INDEX)
    data = [
        {
            "product_id": r["product_id"],
            "name": r["name"],
            "category": r["category"],
            "annual_fee": r["annual_fee"],
            "description": r["description"],
            "embedding": vector_to_bytes(r["embedding"]),
        }
        for r in rows
    ]
    idx.load(data, id_field="product_id")


def load_faqs(client, rows):
    idx = get_index(client, FAQ_INDEX)
    data = [
        {
            "article_id": r["article_id"],
            "category": r["category"],
            "title": r["title"],
            "body": r["body"],
            "embedding": vector_to_bytes(r["embedding"]),
        }
        for r in rows
    ]
    idx.load(data, id_field="article_id")


def _run_vector_query(client, index_name, query_vector, limit, filter_expression=None, return_fields=None):
    idx = get_index(client, index_name)
    q = VectorQuery(
        vector=query_vector,
        vector_field_name="embedding",
        return_fields=return_fields,
        filter_expression=filter_expression,
        num_results=limit,
    )
    t0 = time.perf_counter()
    results = idx.query(q)
    ms = (time.perf_counter() - t0) * 1000
    rows = [
        {**{k: v for k, v in r.items() if k not in ("id", "vector_distance")},
         "score": 1 - float(r["vector_distance"]) / 2}  # cosine distance -> similarity in [0,1]
        for r in results
    ]
    # "id" excluded above is RedisVL's own result key (the full Redis key,
    # e.g. "tx::T00000114") — distinct from our own transaction_id/
    # product_id/article_id schema fields, which are requested explicitly
    # via return_fields and pass straight through untouched.
    return rows, ms, str(q)


# FT.SEARCH tx_flat_idx "*=>[KNN <limit> @embedding $vector]" — exact KNN.
# `vector`, when given, skips the embed() call and searches with it
# directly — bench.py uses this so concurrent-throughput numbers measure
# the datastore, not the (single-threaded, GIL-bound) local embedding
# step every request would otherwise redo.
def vector_search_flat(client, query_text=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    return _run_vector_query(
        client, TX_FLAT_INDEX, vec, limit,
        return_fields=["transaction_id", "merchant", "category", "description", "amount", "date"],
    )


# FT.SEARCH tx_hnsw_idx "*=>[KNN <limit> @embedding $vector]" — approximate
# KNN over an HNSW graph. Same query, same corpus, different index.
def vector_search_hnsw(client, query_text=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    return _run_vector_query(
        client, TX_HNSW_INDEX, vec, limit,
        return_fields=["transaction_id", "merchant", "category", "description", "amount", "date"],
    )


def _build_filter(category=None, max_annual_fee=None):
    expr = None
    if category:
        expr = Tag("category") == category
    if max_annual_fee is not None:
        fee_expr = Num("annual_fee") <= max_annual_fee
        expr = fee_expr if expr is None else (expr & fee_expr)
    return expr


# FT.SEARCH product_idx "(@category:{Travel} @annual_fee:[-inf 100])=>[KNN
# ... ]" — vector search combined with TAG/NUMERIC filters in one query,
# not a post-filter step.
def semantic_search_products(client, query_text=None, category=None, max_annual_fee=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    filt = _build_filter(category, max_annual_fee)
    return _run_vector_query(
        client, PRODUCT_INDEX, vec, limit, filter_expression=filt,
        return_fields=["product_id", "name", "category", "annual_fee", "description"],
    )


def semantic_search_faq(client, query_text=None, category=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    filt = Tag("category") == category if category else None
    return _run_vector_query(
        client, FAQ_INDEX, vec, limit, filter_expression=filt,
        return_fields=["article_id", "category", "title", "body"],
    )


def build_router(client, overwrite=False):
    vectorizer = CustomTextVectorizer(embed=embed, embed_many=embed_many)
    routes = [
        Route(
            name=r["name"],
            references=r["references"],
            distance_threshold=r["distance_threshold"],
            metadata={"description": r["description"], "action": r["action"], "search_target": r["search_target"]},
        )
        for r in ROUTES
    ]
    return SemanticRouter(
        name=ROUTER_NAME,
        routes=routes,
        vectorizer=vectorizer,
        redis_client=client,
        overwrite=overwrite,
    )


def route_query(router, client, query_text):
    # Embed BEFORE starting the timer and call router(vector=...), not
    # router(statement=...) — the latter embeds internally, which would
    # charge Redis's measured time for the same embedding step that
    # pg_store.route_query() already excludes (it embeds before its own
    # t0 too). Measured impact of getting this wrong: ~6ms of embedding
    # time on top of a genuine ~0.7ms Redis round trip — enough on its
    # own to flip who looks faster.
    vec = embed(query_text)
    t0 = time.perf_counter()
    match = router(vector=vec)
    ms = (time.perf_counter() - t0) * 1000
    route = router.get(match.name) if match.name else None
    search_target = route.metadata.get("search_target") if route else None

    # "Route to the right vector search": the SAME query embedding used
    # to classify the intent is reused to search whichever corpus that
    # intent maps to — no second embed() call, no manual "which index"
    # logic in application code. Timed separately from routing itself,
    # not folded into `ms`, so the two costs stay visible rather than
    # conflated into one number.
    search_result = None
    search_ms = None
    if search_target == "products":
        rows, search_ms, _ = semantic_search_products(client, limit=3, vector=vec)
        search_result = rows
    elif search_target == "faq":
        rows, search_ms, _ = semantic_search_faq(client, limit=3, vector=vec)
        search_result = rows

    return {
        "route": match.name,
        "distance": match.distance,
        "description": route.metadata.get("description") if route else None,
        "action": route.metadata.get("action") if route else None,
        "search_target": search_target,
        "search_result": search_result,
        "search_ms": search_ms,
    }, ms


# RedisVL's SemanticCache extension, wrapped around the same shared
# vectorizer as everything else. distance_threshold=0.15 is deliberately
# a bit looser than the library default (0.1) — loose enough to catch an
# obvious paraphrase of a cached question, tight enough not to conflate
# genuinely different questions.
def build_semantic_cache(client, overwrite=False):
    vectorizer = CustomTextVectorizer(embed=embed, embed_many=embed_many)
    return SemanticCache(
        name=CACHE_NAME,
        distance_threshold=CACHE_DISTANCE_THRESHOLD,
        vectorizer=vectorizer,
        redis_client=client,
        overwrite=overwrite,
    )


# Caches transaction search results, not product search — the point of
# this scenario is "repeating an expensive question shouldn't repeat the
# expensive work," and the exact/FLAT transaction search is the
# expensive operation in this demo (Postgres ~15-18ms; see the vector
# search scenario). Caching the 15-row product search would barely be
# visible against either engine's already-sub-millisecond cost there.
#
# Embedding happens BEFORE the timer starts, same convention as every
# other scenario in this file — a cache hit and a cache miss both still
# have to embed the incoming query to know whether anything matches, so
# excluding it here isolates the same thing it isolates everywhere else:
# the datastore-side cost, not a client-side cost identical on both
# engines regardless of caching.
def cached_transaction_search(cache, client, query_text=None, limit=8, vector=None):
    vec = vector if vector is not None else embed(query_text)
    t0 = time.perf_counter()
    hits = cache.check(vector=vec)
    if hits:
        rows = json.loads(hits[0]["response"])
        ms = (time.perf_counter() - t0) * 1000
        distance = float(hits[0].get("vector_distance", 0))
        return rows, ms, True, f"CACHE HIT (distance {distance:.4f} <= {CACHE_DISTANCE_THRESHOLD}) — tx_flat_idx not queried"
    rows, _, query_str = vector_search_flat(client, limit=limit, vector=vec)
    cache.store(prompt=query_text or "(precomputed vector)", response=json.dumps(rows), vector=vec)
    ms = (time.perf_counter() - t0) * 1000
    return rows, ms, False, f"CACHE MISS — computed via {query_str}, stored for next time"


def architecture_info(client):
    info = {}
    for name in (TX_FLAT_INDEX, TX_HNSW_INDEX, PRODUCT_INDEX, FAQ_INDEX, ROUTER_NAME, CACHE_NAME):
        try:
            raw = client.ft(name).info()
            info[name] = {
                "num_docs": int(raw.get("num_docs", 0)),
                "algorithm": _algorithm_of(raw),
            }
        except redis.exceptions.ResponseError:
            info[name] = None
    info["db_size"] = client.dbsize()
    return info


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


def _algorithm_of(ft_info):
    # Our client runs with decode_responses=False (needed for vector byte
    # fields elsewhere), so FT.INFO's nested attribute lists come back as
    # bytes here too — decode before comparing. Matched by field TYPE
    # ("VECTOR"), not by name — our own indexes name their vector field
    # "embedding", but RedisVL's SemanticRouter/SemanticCache extensions
    # name theirs "vector"/"prompt_vector" respectively (see
    # redisvl.extensions.constants). Matching on "embedding" specifically
    # left the Architecture tab reporting "n/a" for intent_router and
    # tx_search_cache even though both really are FLAT-indexed.
    attrs = ft_info.get(b"attributes") or ft_info.get("attributes", [])
    for attr in attrs:
        flat = [_decode(x) for x in attr] if isinstance(attr, list) else []
        pairs = dict(zip(flat[::2], flat[1::2]))
        if pairs.get("type") == "VECTOR":
            return pairs.get("algorithm")
    return None
