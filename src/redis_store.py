"""
Every Redis / RedisVL operation this demo runs, in one place.

Two indexes cover the same request Hashes (`req:<id>`), each reading
the same `embedding` field but built with a different vector algorithm —
`req_flat_idx` (FLAT: exact brute-force KNN, scans every candidate) and
`req_hnsw_idx` (HNSW: approximate nearest-neighbor graph search). Same
data, same field, two independently-built index structures — that's
enough for RediSearch to answer identical queries two different ways, no
duplicated storage required.

`procedure_idx` and `sop_idx` are small (15 and 26 documents) hybrid
indexes — a vector field plus TAG/NUMERIC fields — used for the "look up
the right procedure" and "search the SOP knowledge base" semantic-search
scenarios with filters.

The semantic router uses RedisVL's own `SemanticRouter` extension
untouched, wrapped around the exact same `embeddings.embed`/`embed_many`
functions the rest of this file uses (via `CustomTextVectorizer`), so
"how RedisVL routes a sentence" is not a separately-tuned code path from
"how RedisVL searches a sentence."

`build_semantic_cache()`/`cached_request_search()` use RedisVL's
`SemanticCache` extension the same way — same shared vectorizer, no
separate embedding path.

`check_duplicate()` is the odd one out: it doesn't use RedisVL at all.
Duplicate-identity detection at intake runs two independent Redis
primitives — a `BF.*` Bloom filter for an exact-TIN check, and a plain
`FT.SEARCH` vector KNN for a fuzzy name/city/state check — neither of
which is a RedisVL search scenario like the rest of this file.
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

from config import (
    NEAR_DUPLICATE_DISTANCE_THRESHOLD, REDIS_URL, TIN_BLOOM_CAPACITY,
    TIN_BLOOM_ERROR_RATE, VECTOR_DIM,
)
from embeddings import embed, embed_many
from routes import ROUTES

REQ_FLAT_INDEX = "req_flat_idx"
REQ_HNSW_INDEX = "req_hnsw_idx"
PROCEDURE_INDEX = "procedure_idx"
CUSTOMER_INDEX = "customer_idx"
CACHE_NAME = "req_search_cache"
CACHE_DISTANCE_THRESHOLD = 0.15
SOP_INDEX = "sop_idx"
ROUTER_NAME = "intent_router"
TIN_BLOOM = "tin_seen"


def connect():
    return redis.Redis.from_url(REDIS_URL, decode_responses=False)


# RedisVL's own HNSW defaults are M=16, ef_construction=200 (both fine),
# but ef_runtime=10 — measured directly against this corpus, that's too
# low for good recall: a plain-language query like "address change" (not
# a copy of any stored description) missed its entire target cluster
# and returned unrelated requests instead, 0% overlap with the exact
# result. Raising ef_runtime to 40 fixed it. Don't drop this back to the
# library default on the theory that it's unnecessary tuning — it's the
# difference between a working and a broken HNSW scenario here.
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


# FT.CREATE req_flat_idx ON HASH PREFIX req: SCHEMA id TAG channel TAG
# category TAG days_open NUMERIC embedding VECTOR FLAT ...
# Exact KNN: every candidate vector is compared, no approximation. Correct
# by construction, cost grows linearly with corpus size.
def _req_schema(index_name, algorithm):
    return IndexSchema.from_dict({
        "index": {"name": index_name, "prefix": "req:", "storage_type": "hash"},
        "fields": [
            # NOT named "id" — RedisVL query results always include an "id"
            # key of their own (the full Redis key, e.g. "req::R00000114"),
            # which would collide with and shadow a same-named schema field.
            {"name": "request_id", "type": "tag"},
            {"name": "channel", "type": "tag"},
            {"name": "category", "type": "tag"},
            {"name": "description", "type": "text"},
            {"name": "days_open", "type": "numeric"},
            {"name": "date", "type": "tag"},
            _vector_field(algorithm),
        ],
    })


# FT.CREATE procedure_idx ON HASH PREFIX procedure: SCHEMA ... embedding
# VECTOR FLAT ... — small corpus (15 rows), FLAT vs HNSW makes no
# measurable difference at this scale, so this scenario is about hybrid
# filtering, not algorithm choice.
def _procedure_schema():
    return IndexSchema.from_dict({
        "index": {"name": PROCEDURE_INDEX, "prefix": "procedure:", "storage_type": "hash"},
        "fields": [
            {"name": "procedure_id", "type": "tag"},
            {"name": "name", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "sla_days", "type": "numeric"},
            {"name": "description", "type": "text"},
            _vector_field("flat"),
        ],
    })


def _sop_schema():
    return IndexSchema.from_dict({
        "index": {"name": SOP_INDEX, "prefix": "sop:", "storage_type": "hash"},
        "fields": [
            {"name": "sop_id", "type": "tag"},
            {"name": "category", "type": "tag"},
            {"name": "title", "type": "text"},
            {"name": "body", "type": "text"},
            _vector_field("flat"),
        ],
    })


# FT.CREATE customer_idx ON HASH PREFIX customer: SCHEMA ... embedding
# VECTOR FLAT ... — the fuzzy-match half of duplicate-identity
# detection. `embedding` is over the customer's NAME ONLY, not blended
# with city/state — measured directly (see docs/METHODOLOGY.md) that
# blending city into the text made the false-positive problem worse,
# not better, since two unrelated same-last-name people sharing a city
# token pulled their vectors closer together on top of already sharing
# a last-name token. `city` is filtered on separately as an exact TAG
# match in check_duplicate() instead — hybrid filtering again, same
# pattern as the procedure/SOP semantic search, applied here to narrow
# candidates before ranking rather than to narrow results after.
def _customer_schema():
    return IndexSchema.from_dict({
        "index": {"name": CUSTOMER_INDEX, "prefix": "customer:", "storage_type": "hash"},
        "fields": [
            {"name": "customer_id", "type": "tag"},
            {"name": "name", "type": "text"},
            {"name": "city", "type": "tag"},
            {"name": "state", "type": "tag"},
            {"name": "tin", "type": "tag"},
            _vector_field("flat"),
        ],
    })


def get_index(client, index_name):
    schema = {
        REQ_FLAT_INDEX: _req_schema(REQ_FLAT_INDEX, "flat"),
        REQ_HNSW_INDEX: _req_schema(REQ_HNSW_INDEX, "hnsw"),
        PROCEDURE_INDEX: _procedure_schema(),
        SOP_INDEX: _sop_schema(),
        CUSTOMER_INDEX: _customer_schema(),
    }[index_name]
    return SearchIndex(schema=schema, redis_client=client)


def create_indexes(client):
    # req_hnsw_idx is deliberately NOT created here — see create_hnsw_index()
    # below for why.
    for name in (REQ_FLAT_INDEX, PROCEDURE_INDEX, SOP_INDEX, CUSTOMER_INDEX):
        get_index(client, name).create(overwrite=True, drop=True)


# Create req_hnsw_idx only after load_requests() has already written
# every req: hash. Creating it up front, like the other three indexes, and
# letting RediSearch build the HNSW graph incrementally as each document
# is HSET one at a time (the on-write indexing path) measured
# significantly worse recall than building it here, in one shot, against
# already-loaded data (the same background bulk-scan path RediSearch uses
# when you attach an index to an existing keyspace) — a plain-language
# query with no exact match in the corpus ("address change") missed its
# entire target cluster with the former and found it cleanly with the
# latter, same data, same M/ef_construction/ef_runtime either way. Don't
# move this back to create_indexes().
def create_hnsw_index(client):
    get_index(client, REQ_HNSW_INDEX).create(overwrite=True, drop=True)
    wait_for_indexing(client, REQ_HNSW_INDEX)


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


def load_requests(client, rows):
    """rows: iterable of dicts with an added 'embedding' key (list[float])."""
    idx = get_index(client, REQ_FLAT_INDEX)  # any of the two works, they share the prefix
    data = [
        {
            "request_id": r["request_id"],
            "channel": r["channel"],
            "category": r["category"],
            "description": r["description"],
            "days_open": r["days_open"],
            "date": r["date"],
            "embedding": vector_to_bytes(r["embedding"]),
        }
        for r in rows
    ]
    idx.load(data, id_field="request_id")


def load_procedures(client, rows):
    idx = get_index(client, PROCEDURE_INDEX)
    data = [
        {
            "procedure_id": r["procedure_id"],
            "name": r["name"],
            "category": r["category"],
            "sla_days": r["sla_days"],
            "description": r["description"],
            "embedding": vector_to_bytes(r["embedding"]),
        }
        for r in rows
    ]
    idx.load(data, id_field="procedure_id")


def load_sops(client, rows):
    idx = get_index(client, SOP_INDEX)
    data = [
        {
            "sop_id": r["sop_id"],
            "category": r["category"],
            "title": r["title"],
            "body": r["body"],
            "embedding": vector_to_bytes(r["embedding"]),
        }
        for r in rows
    ]
    idx.load(data, id_field="sop_id")


def load_customers(client, rows):
    idx = get_index(client, CUSTOMER_INDEX)
    data = [
        {
            "customer_id": r["customer_id"],
            "name": r["name"],
            "city": r["city"],
            "state": r["state"],
            "tin": r["tin"],
            "embedding": vector_to_bytes(r["embedding"]),
        }
        for r in rows
    ]
    idx.load(data, id_field="customer_id")


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
    # e.g. "req::R00000114") — distinct from our own request_id/
    # procedure_id/sop_id schema fields, which are requested explicitly
    # via return_fields and pass straight through untouched.
    return rows, ms, str(q)


# FT.SEARCH req_flat_idx "*=>[KNN <limit> @embedding $vector]" — exact KNN.
# `vector`, when given, skips the embed() call and searches with it
# directly — bench.py uses this so concurrent-throughput numbers measure
# the datastore, not the (single-threaded, GIL-bound) local embedding
# step every request would otherwise redo.
def vector_search_flat(client, query_text=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    return _run_vector_query(
        client, REQ_FLAT_INDEX, vec, limit,
        return_fields=["request_id", "channel", "category", "description", "days_open", "date"],
    )


# FT.SEARCH req_hnsw_idx "*=>[KNN <limit> @embedding $vector]" — approximate
# KNN over an HNSW graph. Same query, same corpus, different index.
def vector_search_hnsw(client, query_text=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    return _run_vector_query(
        client, REQ_HNSW_INDEX, vec, limit,
        return_fields=["request_id", "channel", "category", "description", "days_open", "date"],
    )


def _build_filter(category=None, max_sla_days=None):
    expr = None
    if category:
        expr = Tag("category") == category
    if max_sla_days is not None:
        sla_expr = Num("sla_days") <= max_sla_days
        expr = sla_expr if expr is None else (expr & sla_expr)
    return expr


# FT.SEARCH procedure_idx "(@category:{Address Maintenance} @sla_days:[-inf
# 5])=>[KNN ... ]" — vector search combined with TAG/NUMERIC filters in one
# query, not a post-filter step.
def semantic_search_procedures(client, query_text=None, category=None, max_sla_days=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    filt = _build_filter(category, max_sla_days)
    return _run_vector_query(
        client, PROCEDURE_INDEX, vec, limit, filter_expression=filt,
        return_fields=["procedure_id", "name", "category", "sla_days", "description"],
    )


def semantic_search_sop(client, query_text=None, category=None, limit=10, vector=None):
    vec = vector if vector is not None else embed(query_text)
    filt = Tag("category") == category if category else None
    return _run_vector_query(
        client, SOP_INDEX, vec, limit, filter_expression=filt,
        return_fields=["sop_id", "category", "title", "body"],
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
    # charge the measured time with a local, single-threaded embedding
    # step (~6ms on a laptop) that has nothing to do with the actual
    # Redis round trip (~0.7ms). Measure the datastore, not the CPU-bound
    # embedding model.
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
    if search_target == "procedures":
        rows, search_ms, _ = semantic_search_procedures(client, limit=3, vector=vec)
        search_result = rows
    elif search_target == "sop":
        rows, search_ms, _ = semantic_search_sop(client, limit=3, vector=vec)
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


# Caches request search results, not procedure search — the point of
# this scenario is "repeating an expensive question shouldn't repeat the
# expensive work," and the exact/FLAT request search over 40,000 rows is
# the most expensive operation in this demo. Caching the 15-row
# procedure search would barely be visible against its already-
# sub-millisecond cost.
#
# Embedding happens BEFORE the timer starts, same convention as every
# other scenario in this file — a cache hit and a cache miss both still
# have to embed the incoming query to know whether anything matches, so
# excluding it here isolates the same thing it isolates everywhere else:
# the datastore-side cost, not a client-side cost identical on both
# engines regardless of caching.
def cached_request_search(cache, client, query_text=None, limit=8, vector=None):
    vec = vector if vector is not None else embed(query_text)
    t0 = time.perf_counter()
    hits = cache.check(vector=vec)
    if hits:
        rows = json.loads(hits[0]["response"])
        ms = (time.perf_counter() - t0) * 1000
        distance = float(hits[0].get("vector_distance", 0))
        return rows, ms, True, f"CACHE HIT (distance {distance:.4f} <= {CACHE_DISTANCE_THRESHOLD}) — req_flat_idx not queried"
    rows, _, query_str = vector_search_flat(client, limit=limit, vector=vec)
    cache.store(prompt=query_text or "(precomputed vector)", response=json.dumps(rows), vector=vec)
    ms = (time.perf_counter() - t0) * 1000
    return rows, ms, False, f"CACHE MISS — computed via {query_str}, stored for next time"


def tin_exact_key(tin):
    return f"tin:{tin}"


# BF.RESERVE tin_seen 0.01 6000 — a Bloom filter over every TIN in the
# customer base. O(1) and a fraction of a Set's memory, at the cost of
# a small, known false-positive rate — but never a false negative:
# BF.ADD guarantees a BF.EXISTS "no" is always correct, so a miss can
# skip the exact check entirely with zero risk. A "maybe" still needs
# confirming against the real tin:<tin> Set below, since it might be
# one of the false positives BF.RESERVE's error rate allows for.
def build_tin_bloom(client, tins):
    client.delete(TIN_BLOOM)
    client.execute_command("BF.RESERVE", TIN_BLOOM, TIN_BLOOM_ERROR_RATE, TIN_BLOOM_CAPACITY)
    for i in range(0, len(tins), 1000):
        client.execute_command("BF.MADD", TIN_BLOOM, *tins[i : i + 1000])


def load_tin_index(client, rows):
    pipe = client.pipeline(transaction=False)
    for r in rows:
        pipe.sadd(tin_exact_key(r["tin"]), r["customer_id"])
    pipe.execute()


# The duplicate-detection scenario: two independent signals, neither of
# which is a RedisVL search.
#
#   1. BF.EXISTS tin_seen <tin> — O(1). A "no" is a hard guarantee of
#      no existing customer with this exact TIN, decided without ever
#      touching the real per-TIN data.
#   2. On a "maybe", SMEMBERS tin:<tin> confirms it for real — this is
#      also where a Bloom false positive would get caught
#      (maybe_seen=True, exact_matches=[]).
#   3. Independent of both: a fuzzy FT.SEARCH KNN over customer_idx on
#      "<name>, <city>, <state>" catches near-duplicate profiles that
#      don't share an exact TIN at all — a middle initial, a suffix,
#      the same person entered twice under slightly different details.
def check_duplicate(client, name, city, state, tin):
    t0 = time.perf_counter()
    maybe_seen = bool(client.execute_command("BF.EXISTS", TIN_BLOOM, tin))
    bloom_ms = (time.perf_counter() - t0) * 1000

    exact_matches, exact_ms = [], None
    if maybe_seen:
        t1 = time.perf_counter()
        members = client.smembers(tin_exact_key(tin))
        exact_ms = (time.perf_counter() - t1) * 1000
        exact_matches = sorted(m.decode() for m in members)

    vec = embed(name)
    fuzzy_rows, fuzzy_ms, _ = _run_vector_query(
        client, CUSTOMER_INDEX, vec, limit=3, filter_expression=Tag("city") == city,
        return_fields=["customer_id", "name", "city", "state", "tin"],
    )
    # `score` from _run_vector_query is similarity (1 - distance/2);
    # convert back to cosine distance to compare against the same
    # threshold generate.py's near_duplicate_name() was tuned against.
    for row in fuzzy_rows:
        row["near_duplicate"] = (2 * (1 - row["score"])) <= NEAR_DUPLICATE_DISTANCE_THRESHOLD

    return {
        "maybe_seen": maybe_seen,
        "bloom_ms": bloom_ms,
        "exact_matches": exact_matches,
        "exact_ms": exact_ms,
        "false_positive": maybe_seen and not exact_matches,
        "fuzzy_matches": fuzzy_rows,
        "fuzzy_ms": fuzzy_ms,
    }


def _bloom_info(client, name):
    try:
        raw = client.execute_command("BF.INFO", name)
    except redis.exceptions.ResponseError:
        return None
    pairs = dict(zip((_decode(x) for x in raw[::2]), (_decode(x) for x in raw[1::2])))
    return {
        "capacity": pairs.get("Capacity"),
        "size_bytes": pairs.get("Size"),
        "items_inserted": pairs.get("Number of items inserted"),
    }


def architecture_info(client):
    info = {}
    for name in (REQ_FLAT_INDEX, REQ_HNSW_INDEX, PROCEDURE_INDEX, SOP_INDEX, CUSTOMER_INDEX, ROUTER_NAME, CACHE_NAME):
        try:
            raw = client.ft(name).info()
            info[name] = {
                "num_docs": int(raw.get("num_docs", 0)),
                "algorithm": _algorithm_of(raw),
            }
        except redis.exceptions.ResponseError:
            info[name] = None
    info[TIN_BLOOM] = _bloom_info(client, TIN_BLOOM)
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
    # req_search_cache even though both really are FLAT-indexed.
    attrs = ft_info.get(b"attributes") or ft_info.get("attributes", [])
    for attr in attrs:
        flat = [_decode(x) for x in attr] if isinstance(attr, list) else []
        pairs = dict(zip(flat[::2], flat[1::2]))
        if pairs.get("type") == "VECTOR":
            return pairs.get("algorithm")
    return None
