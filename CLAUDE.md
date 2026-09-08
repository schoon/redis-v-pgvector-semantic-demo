# redis-v-pgvector-semantic-demo

A demo comparing Redis (RedisVL) against Postgres (pgvector) for vector
search, hybrid semantic search, semantic caching, and semantic intent
routing (including routing straight into the right downstream search),
over a fictitious bank's customer identity operations: the requests that
keep customer identity records accurate and compliant (address updates,
duplicate TIN/SSN resolution, and related customer-data maintenance and
due-diligence work), a procedure catalog, and a support SOP knowledge
base. Python/FastAPI backend, single-file vanilla frontend, both engines
in Docker.

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

**Routing doesn't stop at naming an intent — it searches the right
corpus with the SAME embedding, on both engines.** Each route in
`routes.py` has a `search_target` (`"procedures"`, `"sop"`, or `None`).
`route_query()` on both sides reuses the query vector it already
computed for classification to immediately search that corpus — no
second `embed()` call, no manual "which index" branching in application
code. Verified directly: for `"what's the process for a duplicate TIN
case"`, both engines route to `procedure_lookup` and return the
identical top-3 procedures in the same order.

**Semantic caching has no pgvector-side equivalent, and that's the
point, not a gap to fill.** `pg_store.py` doesn't get a hand-rolled
cache to keep things "fair" — a fair comparison here means Postgres
does exactly what it would really do: recompute every time, since it
has no caching primitive. `cached_request_search()` (not
`cached_procedure_search()` — see docs/METHODOLOGY.md for why the
target changed) caches the *expensive* exact-search scenario, not the
trivially cheap 15-row procedure search, so the win is real and visible
rather than sub-millisecond and easy to dismiss.

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

## Real bugs, all load-bearing fixes — don't undo them

**`req_hnsw_idx` must be created AFTER `load_requests()`, never
before.** Building the HNSW index up front and letting RediSearch add
each document to the graph one at a time as it's `HSET` (the on-write
indexing path) measured significantly worse recall than creating the
index against already-loaded data (RediSearch's background bulk-scan
path) — same final data, same `m`/`ef_construction`/`ef_runtime` either
way. Concretely: the query `"address change"` (no exact match in the
corpus) returned unrelated requests with the incremental build and
correctly returned "Address Update" requests with the bulk-scan build.
`redis_store.create_indexes()` deliberately excludes `req_hnsw_idx`;
`seed_redis.py` calls `create_hnsw_index()` — which also waits on
`FT.INFO`'s `percent_indexed` reaching 1.0, since the bulk scan is
async — only after `load_requests()` returns. If you add a new
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

**Postgres's cursor-per-call inefficiency.** `pg_store.py`'s hot-path
query functions cache a single cursor on the connection object
(`_cursor()`) rather than opening a fresh one with `with client.cursor()
as cur:` on every call — measured directly, that costs ~25% more for a
small query (0.24ms vs 0.18ms). This makes Postgres *faster*, not closer
to Redis; fixed anyway, since leaving a competitor's inefficiency in
place to look better by comparison isn't the kind of fair fight this
demo is for.

**`redis_store.route_query()` must embed BEFORE starting its timer and
call `router(vector=vec)`, never `router(statement=query_text)`.** The
latter embeds internally, charging Redis's measured time for a ~6ms
local embedding step that `pg_store.route_query()`'s measurement already
excludes (it always embedded before its own timer too). This single
line made routing look like a 5× Postgres win when the actual Redis
round trip was faster than Postgres's the whole time — profiled and
confirmed via `embed()`-alone vs `router(vector=...)`-alone timing
before fixing. See docs/METHODOLOGY.md for the full profile. If you
touch either `route_query()`, re-time embed-only vs. round-trip-only
before trusting the combined number — this bug produced numbers that
looked plausible, not obviously broken.

## Other things not to "clean up"

**Watch for doubled backslashes (`\\n`, `\\'`) in `public/index.html` —
they've broken this file twice.** A `\\n` where JS needs `\n` doesn't
error; it silently produces the two visible characters `\n` instead of
a line break (found in `runArchitecture()`'s index list — this is what
made "see query pane" look like garbled text instead of a readable
list). A `\\'` where JS needs `\'` is worse — it terminates the string
early and throws a `SyntaxError` that silently breaks every tab's
rendering (found once already in this file). If you hand-write escaped
quotes or newlines into this file, grep for `\\\\[nrt'"]` afterward and
load the page in a browser — a passing `ast.parse`/lint pass in Python
files doesn't catch either failure mode, since both are perfectly valid
JS strings that just aren't the string you meant.

**RedisVL query results always include an `id` key of their own — the
full Redis key (e.g. `req::R00000114`) — regardless of `return_fields`.**
This is why every schema in `redis_store.py` names its own identifier
field `request_id`/`procedure_id`/`sop_id`, never `id`: a same-named
schema field would collide with and be shadowed by RedisVL's own `id`.
If you add a new index, follow the same naming, and don't reintroduce a
field literally named `id`.

**The verdict pill in `public/index.html` (`setVerdict()`) computes its
winner and ratio directly from the two on-screen `ms`/`qps` values,
never from a server-supplied ratio.** A sibling demo in this series
shipped a version that assumed one engine always wins and rendered a
nonsensical result once that stopped being true. This demo's numbers
aren't uniform either — HNSW search and routing are roughly even at
this corpus's scale, sometimes landing on either side by a small margin
(see docs/METHODOLOGY.md) — so `setVerdict()` has to get the direction
right every time, not most of the time.

**Recall numbers near 0% on the HNSW tab, with an otherwise-correct
category, are a tie-breaking artifact, not a bug.** Many requests share
literally identical description text (and so identical embeddings) —
"top-8 by exact request ID" is ambiguous whenever more than 8 rows tie
at the k-th smallest distance. `validate.py` checks correctness by
distance value against a tie-aware threshold, not by exact ID-set
equality, for exactly this reason — see `is_valid_top_k()`.

**`bench.py` embeds its query vector once, outside the timed loop, and
passes it directly via `vector=` to both stores' `vector_search_flat()`
— never re-embeds per request.** Re-embedding inside the timed loop
would make concurrent "throughput" mostly measure how fast one CPU core
can run the embedding model under Python's GIL, not either datastore.

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
  keys, built from `routes.py`'s `ROUTES`, untouched by this repo's code
  beyond passing in a `CustomTextVectorizer`. Each `Route`'s `metadata`
  dict carries `description`/`action`/`search_target` — client-side only,
  never written to Redis (RedisVL's `_add_routes()` doesn't persist it),
  which is why `route_query()` reads it off `router.get(name).metadata`
  rather than expecting to fetch it back from a hash field.
- `req_search_cache` — RedisVL `SemanticCache`'s own index and entry
  keys. Cleared at the start of every `/api/semantic-cache` call so the
  "first call" in that demo is always a genuine miss.

No real customer or case data is loaded into either engine — see
"What this demo does not support" in docs/METHODOLOGY.md for why.

## Postgres schema

`requests`, `procedures`, `sop_articles` — each with an `embedding
vector(384)` column. `requests` additionally has an HNSW index (`USING
hnsw (embedding vector_cosine_ops)`, built after the bulk load) for the
approximate-search scenario; the exact-search scenario disables index
scans for that one query (`SET LOCAL enable_indexscan/
enable_bitmapscan = off`) rather than using a second table or column.
`routes`/`route_references` hold the hand-rolled router's data — same
`ROUTES` definition as `redis_store.py`'s `SemanticRouter`, loaded by
`pg_store.load_routes()`. `routes.search_target` is a real column (not
client-side-only like Redis's route metadata), read back in
`pg_store.route_query()`'s own SQL to decide whether to run a follow-up
corpus search. There is no cache table — pgvector's side of the
semantic-caching scenario is just `vector_search_flat()`, called fresh
every time.

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
