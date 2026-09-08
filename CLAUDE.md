# redis-v-pgvector-semantic-demo

A showcase of Redis (RedisVL) as the search engine for a fictitious
bank's customer identity operations: the requests that keep customer
identity records accurate and compliant (address updates, duplicate
TIN/SSN resolution, and related customer-data maintenance and
due-diligence work), a procedure catalog, and a support SOP knowledge
base. Each of the seven tabs in the UI shows one Redis/RedisVL search
capability — exact vector search, approximate vector search, hybrid
semantic search, semantic caching, semantic routing, concurrent
throughput, and live schema introspection — plus a persistent
explanation of exactly which Redis command or RedisVL feature is
producing that result. Python/FastAPI backend, single-file vanilla
frontend, Redis in Docker.

Published at
<https://github.com/schoon/redis-v-pgvector-semantic-demo> (private).

This repo used to compare Redis against Postgres/pgvector side by side
(hence the repo name) — that comparison was deliberately removed. The
goal now is to show *how* Redis powers each kind of search, not to
race it against anything. If you're tempted to bring a second engine
back in, don't just bolt it onto the existing endpoints — the whole
frontend and API shape were redesigned around a single result, not a
pair, and re-introducing a second engine without redesigning both back
would leave the two half-matched.

## The one rule that matters: every claim in the UI must be true right now

This is vendor-authored material for a fictitious but plausible use
case. If the "explain" panel on a tab describes a command or mechanism
that isn't actually what's running, the demo is worse than useless —
it teaches something false about Redis. Keep the explain copy in
`public/index.html` in sync with the real command shape in
`redis_store.py` whenever either changes.

**Semantic caching and semantic routing are shown as genuine
capabilities, not just fast queries.** `redis_store.py`'s
`build_semantic_cache()`/`cached_request_search()` and
`build_router()`/`route_query()` wrap RedisVL's `SemanticCache` and
`SemanticRouter` extensions directly — no hand-rolled cosine-distance
math standing in for what the library already does. The point of both
tabs is "here's a capability a plain vector index doesn't hand you for
free," not "here's a faster way to run the same query."

**Routing doesn't stop at naming an intent — it searches the right
corpus with the SAME embedding.** Each route in `routes.py` has a
`search_target` (`"procedures"`, `"sop"`, or `None`). `route_query()`
reuses the query vector it already computed for classification to
immediately search that corpus — no second `embed()` call, no manual
"which index" branching in application code. Verified directly: for
`"what's the process for a duplicate TIN case"`, routing lands on
`procedure_lookup` and returns the correct top-3 procedures against
independently-computed brute-force ground truth (see `validate.py`).

## Stack

- **Backend:** Python 3.12, FastAPI + Uvicorn.
- **Redis client:** [`redisvl`](https://github.com/redis/redis-vl-python)
  (RedisVL) — `SearchIndex`, `VectorQuery`, `FilterQuery`,
  `SemanticRouter`, `SemanticCache`. Vectorizer is a
  `CustomTextVectorizer` wrapping `embeddings.py`, not RedisVL's own HF
  vectorizer, so every scenario embeds identically.
- **Embeddings:** `sentence-transformers`, `all-MiniLM-L6-v2`, 384
  dimensions, CPU, local — no API key, no network at query time.
- **Frontend:** one `public/index.html`. No framework, no build step.

## Commands

```bash
docker compose up -d       # Redis 8 on :6384
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python src/generate.py     # synthetic corpus -> data/*.jsonl
python src/seed_redis.py   # embed + load Redis (indexes + router + cache)
python src/server.py       # demo on :8040
python src/bench.py        # concurrent throughput -> data/bench-results.json
python src/validate.py     # independent ground-truth checks
```

`seed_redis.py` is safe to re-run; it flushes Redis first.

## Real bugs, all load-bearing fixes — don't undo them

**`req_hnsw_idx` must be created AFTER `load_requests()`, never
before.** Building the HNSW index up front and letting RediSearch add
each document to the graph one at a time as it's `HSET` (the on-write
indexing path) measured significantly worse recall than creating the
index against already-loaded data (RediSearch's background bulk-scan
path) — same final data, same `m`/`ef_construction`/`ef_runtime`
either way. Concretely: the query `"address change"` (no exact match
in the corpus) returned unrelated requests with the incremental build
and correctly returned "Address Update" requests with the bulk-scan
build. `redis_store.create_indexes()` deliberately excludes
`req_hnsw_idx`; `seed_redis.py` calls `create_hnsw_index()` — which
also waits on `FT.INFO`'s `percent_indexed` reaching 1.0, since the
bulk scan is async — only after `load_requests()` returns. If you add
a new HNSW-indexed field, follow the same order.

**RedisVL's default HNSW `ef_runtime` (10) is too low for good recall
on this corpus.** Even with the correct build order above, a low
query-time candidate list size can still miss a large, well-separated
cluster for a query with no exact match in the corpus. Raised to 40 in
`redis_store.py`'s `HNSW_EF_RUNTIME`. Don't drop this back to the
library default; measure recall again if you do (the Vector search —
approximate tab's "recall vs Redis's own exact result" number is
exactly this check, live).

**`redis_store.route_query()` must embed BEFORE starting its timer and
call `router(vector=vec)`, never `router(statement=query_text)`.** The
latter embeds internally, which would fold a ~6ms local, single-
threaded embedding step into what's supposed to be a measurement of
the Redis round trip alone (~0.7ms on a laptop) — enough to make the
number look nothing like what it's claiming to measure. If you touch
`route_query()`, re-time embed-only vs. round-trip-only before trusting
the combined number.

## Other things not to "clean up"

**Watch for doubled backslashes (`\\n`, `\\'`) in `public/index.html` —
they've broken this file twice.** A `\\n` where JS needs `\n` doesn't
error; it silently produces the two visible characters `\n` instead of
a line break. A `\\'` where JS needs `\'` is worse — it terminates the
string early and throws a `SyntaxError` that silently breaks every
tab's rendering (found once already in this file). If you hand-write
escaped quotes or newlines into this file, grep for `\\\\[nrt'"]`
afterward and load the page in a browser — a passing `ast.parse`/lint
pass in Python files doesn't catch either failure mode, since both are
perfectly valid JS strings that just aren't the string you meant.

**RedisVL query results always include an `id` key of their own — the
full Redis key (e.g. `req::R00000114`) — regardless of
`return_fields`.** This is why every schema in `redis_store.py` names
its own identifier field `request_id`/`procedure_id`/`sop_id`, never
`id`: a same-named schema field would collide with and be shadowed by
RedisVL's own `id`. If you add a new index, follow the same naming, and
don't reintroduce a field literally named `id`.

**Recall numbers near 0% on the HNSW tab, with an otherwise-correct
category, are a tie-breaking artifact, not a bug.** Many requests
share literally identical description text (and so identical
embeddings) — "top-8 by exact request ID" is ambiguous whenever more
than 8 rows tie at the k-th smallest distance. `validate.py` checks
correctness by distance value against a tie-aware threshold, not by
exact ID-set equality, for exactly this reason — see
`is_valid_top_k()`.

**`bench.py` embeds its query vector once, outside the timed loop, and
passes it directly via `vector=` to `vector_search_flat()` — never
re-embeds per request.** Re-embedding inside the timed loop would make
concurrent "throughput" mostly measure how fast one CPU core can run
the embedding model under Python's GIL, not Redis.

## Redis data model

- `req:<request_id>` — Hash. `channel`, `category`, `description`
  (TEXT), `days_open`, `date`, `embedding` (VECTOR, 40,000-row sample
  only). Indexed by **two** independent indexes reading the same
  `embedding` field: `req_flat_idx` (FLAT/exact) and `req_hnsw_idx`
  (HNSW/approximate).
- `procedure:<procedure_id>` — Hash, indexed by `procedure_idx` (FLAT +
  TAG/NUMERIC hybrid filtering).
- `sop:<sop_id>` — Hash, indexed by `sop_idx` (FLAT + TAG filtering).
- `intent_router` — RedisVL `SemanticRouter`'s own index and reference
  keys, built from `routes.py`'s `ROUTES`. Each `Route`'s `metadata`
  dict carries `description`/`action`/`search_target` — client-side
  only, never written to Redis (RedisVL's `_add_routes()` doesn't
  persist it), which is why `route_query()` reads it off
  `router.get(name).metadata` rather than expecting to fetch it back
  from a hash field.
- `req_search_cache` — RedisVL `SemanticCache`'s own index and entry
  keys. Cleared at the start of every `/api/semantic-cache` call so the
  "first call" in that demo is always a genuine miss.

No real customer or case data is loaded — see "What this demo does not
support" in docs/METHODOLOGY.md for why.

## Layout

```
docker-compose.yml
requirements.txt
src/
  config.py          scale constants, ports, connection strings
  generate.py         synthetic corpus -> data/*.jsonl
  embeddings.py         the one place text becomes a vector
  routes.py               the one definition of "which intent is this"
  redis_store.py           every RedisVL operation
  seed_redis.py              bulk loader + index builder
  server.py                    FastAPI routes
  bench.py                      concurrent throughput CLI
  validate.py                    independent ground-truth checks
public/
  index.html                       the whole frontend
data/                                generated, gitignored
docs/
  METHODOLOGY.md
```

Keep the frontend a single self-contained `index.html` — no build step.
