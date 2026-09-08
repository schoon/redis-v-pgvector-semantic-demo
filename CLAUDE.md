# redis-v-pgvector-semantic-demo

A demo comparing Redis (RedisVL) against Postgres (pgvector) for vector
search, hybrid semantic search, and semantic intent routing, over a
fictitious credit-card company's transactions, card products, and
support FAQ. Python/FastAPI backend, single-file vanilla frontend, both
engines in Docker.

Published at
<https://github.com/schoon/redis-v-pgvector-semantic-demo> (private).

## The one rule that matters: the comparison must stay fair

This is vendor-authored competitive material. If the two engines are
found to be answering different questions, or one side is quietly
crippled to make a point, the demo is worse than useless — it costs
credibility with the audience it's meant to persuade.

**Both engines embed with the exact same model, the exact same way** —
`embeddings.py` is the only place text becomes a vector, and both
`redis_store.py`/`seed_redis.py` and `pg_store.py`/`seed_pg.py` import it
rather than rolling their own. A latency or ranking difference is a
data-access difference, never an embedding difference.

**Both engines get comparable configuration, not sandbagged
configuration.** `docker-compose.yml`'s Postgres tuning (`shared_buffers`,
`effective_cache_size`) is sized up, not down, and nothing disables
Postgres's query planner or parallelism to make a point. Where a choice
has to be made (HNSW `m`/`ef_construction`/`ef_search` vs `ef_runtime`),
the two engines get equivalent values — see "Two real bugs" below for
why that specifically mattered here.

**Semantic routing runs the identical algorithm on both engines, not two
similar ones.** `pg_store.py`'s hand-rolled router doesn't just "average
distance to references" — it replicates RedisVL `SemanticRouter`'s exact
two-stage filter (drop references beyond the global max threshold,
average only the survivors, accept only if under the winning route's own
threshold). An earlier version skipped the first stage and could pick a
different winner than RedisVL would for the same sentence. Verified
directly: for five different test sentences, both engines return the
same route with the same distance to 6 decimal places. If you touch
`pg_store.route_query()`, re-verify against `redis_store.route_query()`
for the same sentences before assuming a simplification is equivalent.

## Stack

- **Backend:** Python 3.12, FastAPI + Uvicorn.
- **Redis client:** [`redisvl`](https://github.com/redis/redis-vl-python)
  (RedisVL) — `SearchIndex`, `VectorQuery`, `FilterQuery`,
  `SemanticRouter`. Vectorizer is a `CustomTextVectorizer` wrapping
  `embeddings.py`, not RedisVL's own HF vectorizer, so the router and
  every search scenario embed identically.
- **Postgres:** `psycopg` (v3) + the `pgvector` Python package for the
  `vector` type adapter. Plain SQL, no ORM.
- **Embeddings:** `sentence-transformers`, `all-MiniLM-L6-v2`, 384
  dimensions, CPU, local — no API key, no network at query time.
- **Frontend:** one `public/index.html`. No framework, no build step.

## Commands

```bash
docker compose up -d       # Redis 8 on :6384, Postgres+pgvector on :5433
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python src/generate.py     # synthetic corpus -> data/*.jsonl
python src/seed_redis.py   # embed + load Redis (indexes + router)
python src/seed_pg.py      # embed + load Postgres (tables + HNSW index)
python src/server.py       # demo on :8040
python src/bench.py        # concurrent throughput -> data/bench-results.json
python src/validate.py     # independent ground-truth checks
```

`seed_redis.py`/`seed_pg.py` are safe to re-run; both wipe their engine
first.

## Two real bugs, both load-bearing fixes — don't undo them

**`tx_hnsw_idx` must be created AFTER `load_transactions()`, never
before.** Building the HNSW index up front and letting RediSearch add
each document to the graph one at a time as it's `HSET` (the on-write
indexing path) measured significantly worse recall than creating the
index against already-loaded data (RediSearch's background bulk-scan
path) — same final data, same `m`/`ef_construction`/`ef_runtime` either
way. Concretely: the query `"coffee purchase"` (no exact match in the
corpus) returned "Online Shopping" transactions with the incremental
build and correctly returned "Coffee Shops" with the bulk-scan build.
`redis_store.create_indexes()` deliberately excludes `tx_hnsw_idx`;
`seed_redis.py` calls `create_hnsw_index()` — which also waits on
`FT.INFO`'s `percent_indexed` reaching 1.0, since the bulk scan is
async — only after `load_transactions()` returns. If you add a new
HNSW-indexed field, follow the same order, and follow pgvector's own
documented advice on the Postgres side (`create_hnsw_index()` there is
already called after the bulk load, not before).

**RedisVL's default HNSW `ef_runtime` (10) is too low for good recall on
this corpus.** Even with the correct build order above, a low
query-time candidate list size can still miss a large, well-separated
cluster for a query with no exact match in the corpus. Raised to 40 in
`redis_store.py`'s `HNSW_EF_RUNTIME`, matching pgvector's
`HNSW_EF_SEARCH = 40` in `pg_store.py` — chosen to be comparable on both
engines, not to flatter either one. Don't drop this back to the library
default; measure recall again if you do.

## Other things not to "clean up"

**RedisVL query results always include an `id` key of their own — the
full Redis key (e.g. `tx::T00000114`) — regardless of `return_fields`.**
This is why every schema in `redis_store.py` names its own identifier
field `transaction_id`/`product_id`/`article_id`, never `id`: a
same-named schema field would collide with and be shadowed by RedisVL's
own `id`. If you add a new index, follow the same naming, and don't
reintroduce a field literally named `id`.

**The verdict pill in `public/index.html` (`setVerdict()`) computes its
winner and ratio directly from the two on-screen `ms`/`qps` values,
never from a server-supplied ratio.** A sibling demo in this series
shipped a version that assumed one engine always wins and rendered a
nonsensical result once that stopped being true. This demo's numbers are
genuinely mixed — Postgres wins on HNSW search and on routing at this
corpus's scale (see docs/METHODOLOGY.md) — so `setVerdict()` has to get
the direction right every time, not most of the time.

**Recall numbers near 0% on the HNSW tab, with an otherwise-correct
category, are a tie-breaking artifact, not a bug.** Many transactions
share literally identical description text (and so identical
embeddings) — "top-8 by exact transaction ID" is ambiguous whenever more
than 8 rows tie at the k-th smallest distance. `validate.py` checks
correctness by distance value against a tie-aware threshold, not by
exact ID-set equality, for exactly this reason — see
`is_valid_top_k()`.

**`bench.py` embeds its query vector once, outside the timed loop, and
passes it directly via `vector=` to both stores' `vector_search_flat()`
— never re-embeds per request.** Re-embedding inside the timed loop
would make concurrent "throughput" mostly measure how fast one CPU core
can run the embedding model under Python's GIL, not either datastore.

## Redis data model

- `tx:<transaction_id>` — Hash. `merchant`, `category`, `description`
  (TEXT), `amount`, `date`, `embedding` (VECTOR, 40,000-row sample only).
  Indexed by **two** independent indexes reading the same `embedding`
  field: `tx_flat_idx` (FLAT/exact) and `tx_hnsw_idx` (HNSW/approximate).
- `product:<product_id>` — Hash, indexed by `product_idx` (FLAT +
  TAG/NUMERIC hybrid filtering).
- `faq:<article_id>` — Hash, indexed by `faq_idx` (FLAT + TAG filtering).
- `intent_router` — RedisVL `SemanticRouter`'s own index and reference
  keys, built from `routes.py`'s `ROUTES`, untouched by this repo's code
  beyond passing in a `CustomTextVectorizer`.

No customer/card data is loaded into either engine — see
"What this demo does not support" in docs/METHODOLOGY.md for why.

## Postgres schema

`transactions`, `card_products`, `faq_articles` — each with an
`embedding vector(384)` column. `transactions` additionally has an HNSW
index (`USING hnsw (embedding vector_cosine_ops)`, built after the bulk
load) for the approximate-search scenario; the exact-search scenario
disables index scans for that one query (`SET LOCAL enable_indexscan/
enable_bitmapscan = off`) rather than using a second table or column.
`routes`/`route_references` hold the hand-rolled router's data — same
`ROUTES` definition as `redis_store.py`'s `SemanticRouter`, loaded by
`pg_store.load_routes()`.

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
  pg_store.py               every Postgres/pgvector operation
  seed_redis.py              bulk loader + index builder
  seed_pg.py                  bulk loader + index builder
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
