# redis-v-pgvector-semantic-demo

Side-by-side **vector search**, **hybrid semantic search**, **semantic
caching**, and **semantic intent routing** — **Redis (RedisVL)** vs
**Postgres (pgvector)** — over a fictitious bank's customer identity
operations: the requests that keep customer identity records accurate
and compliant (address updates, duplicate TIN/SSN resolution, and
related customer-data maintenance and due-diligence work), a procedure
catalog, and a support SOP knowledge base, with per-query latency shown
live.

> **Method and caveats live in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).**
> Read it before presenting — it covers two real HNSW bugs found while
> building this (building the index before the bulk load, and RedisVL's
> default query-time candidate count being too low for this corpus), a
> real timing bug that made routing look far slower than it is (embedding
> the query inside vs. outside the timed call), why semantic routing runs
> the *identical* classification algorithm on both engines rather than
> two similar ones, why semantic caching has no pgvector-side equivalent
> to compare against feature-for-feature, and what this demo does not
> support (including a newer Redis primitive — Vector Sets — considered
> and deliberately not used here).

No real customer or case data of any kind — every ID, name, and number
in the corpus is synthetic. See
[Methodology](docs/METHODOLOGY.md#no-real-customer-data-anywhere).

## Quick start

```bash
docker compose up -d
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/generate.py
.venv/bin/python src/seed_redis.py
.venv/bin/python src/seed_pg.py
.venv/bin/python src/server.py
```

Then open **<http://localhost:8040>**.

### Before you start

| Need | Why |
| ---- | --- |
| **Docker** with Compose v2 | runs Redis and Postgres+pgvector |
| **Python 3.12** | torch/sentence-transformers wheels; newer Pythons may not have them yet |
| **~2 GB free disk** | generated data + the embedding model (downloaded once, cached) |
| Ports **8040**, **6384**, **5433** free | app, Redis, Postgres |

First run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)
from Hugging Face and caches it under `~/.cache/huggingface` — no API
key, no network needed on later runs.

### Stopping

```bash
docker compose down
```

No volumes are declared, so this discards the data. Re-run
`seed_redis.py`/`seed_pg.py` any time — both wipe their engine first.

## The scenarios

| Scenario | Redis | Postgres |
| -------- | ----- | -------- |
| **Vector search — exact (FLAT/KNN)** | `FT.SEARCH` FLAT algorithm | sequential scan (HNSW index scans disabled for this query) |
| **Vector search — approximate (HNSW)** | `FT.SEARCH` HNSW algorithm | `ORDER BY embedding <=> query` via the HNSW index |
| **Semantic search** | `FT.SEARCH` vector + TAG/NUMERIC filter, one query | `WHERE` + `ORDER BY embedding <=>`, one query |
| **Semantic caching** | RedisVL `SemanticCache` (built-in extension) — repeat/similar questions hit the cache, skip the search entirely | no caching primitive — every call re-runs the full search |
| **Semantic routing** | RedisVL `SemanticRouter` (built-in extension); a matched intent immediately reuses the same query embedding to search the right corpus | hand-rolled SQL replicating the identical classification, then the same corpus search |
| **Concurrent throughput** | `python src/bench.py` | `python src/bench.py` |
| **Architecture** | live schema introspection | live schema introspection |

Both engines embed every query with the exact same local model
(`src/embeddings.py`) — a latency or ranking difference is a
data-access difference, never an embedding difference. Every scenario
reaches the **same result set** on both engines for the same query,
verified against independently-computed ground truth — see
[Validating the results](#validating-the-results).

## Observed on one laptop

**Measured 2026-09-08.** 40,000 embedded requests, 15 procedures,
26 SOP articles, 87 routing reference utterances. Redis 8 and Postgres
17 + pgvector 0.8.6, both in Docker on a 14-core Apple-silicon MacBook.
Median of 3 runs per sample.

| Scenario | Redis | Postgres | Winner |
| -------- | ----- | -------- | ------ |
| Vector search (exact) | 4.7–8.1 ms | 17.8–19.7 ms | **Redis, ~2–4×** |
| Vector search (HNSW) | 0.9–1.8 ms | 0.4–0.9 ms | roughly even |
| Semantic search — procedures | 0.7–0.9 ms | 0.5–0.7 ms | roughly even |
| Semantic search — SOP | 0.7–1.1 ms | 0.4–1.0 ms | roughly even |
| Semantic caching — first call (miss) | 4.1–7.9 ms | 15.2–25.4 ms | **Redis, ~2–6×** |
| Semantic caching — repeat calls (avg) | 0.7–0.8 ms | 15.7–18.7 ms | **Redis, ~19–28×** |
| Semantic routing | 0.8–0.9 ms | 0.7–1.1 ms | roughly even |

**A real measurement bug used to make routing look much worse than it
is — worth naming, not just quietly fixing.** An earlier version of
`redis_store.route_query()` called RedisVL's `router(statement=query)`,
which embeds the query text *inside* the timed call; `pg_store.py`'s
equivalent embeds *before* starting its own timer. That's not a small
difference — local embedding took ~6ms on this laptop, dwarfing the
actual ~0.7ms Redis round trip it was being added to, and made routing
look like a 5× Postgres win. Embedding the query first and calling
`router(vector=...)` (matching what `pg_store.py` already did) turned
that into the roughly-even result above. See
[Methodology](docs/METHODOLOGY.md) for the full writeup — this is
exactly the kind of asymmetric-timing bug the "both engines answer the
same question" discipline in CLAUDE.md exists to catch, and it had been
sitting in the numbers unnoticed until directly profiled.

**HNSW is a genuine, small, roughly-even result, not a bug.** Both
engines' HNSW result matches their own exact result's score exactly
(verified — see Methodology); the *specific* request IDs returned
can differ because many requests share identical text and tie
exactly on distance, and each engine breaks ties in its own
deterministic order. Whichever engine is faster by a few tenths of a
millisecond at this row count isn't a meaningful signal either way —
both are comfortably sub-2ms.

**Semantic caching is the widest, clearest win in this demo, and it's
architectural, not tuning.** pgvector has no caching primitive at all —
`pg_store.py`'s side of that scenario is just the plain exact search,
called fresh every time, because there is nothing else to call. RedisVL's
`SemanticCache` means a repeated or paraphrased question never touches
`req_flat_idx` again after the first time it's asked. For a corpus this
size the gap is already 18–20× on repeat traffic; the real story it's
standing in for — caching an *expensive* operation (a large-corpus
search, an LLM call) rather than a cheap 40,000-row one — would make
the gap larger, not smaller, since the miss cost that's being avoided
only grows.

**Redis's clearest wins are semantic caching, exact vector search over
the full 40,000-row sample, and concurrent throughput under load** — all
shown below. At this laptop-sized scale, HNSW and hybrid semantic search
are close enough that the honest headline there is "comparable," not
"Redis wins everything" — see Methodology for why, and don't smooth that
over when presenting this.

## Concurrent throughput

```bash
.venv/bin/python src/bench.py
.venv/bin/python src/bench.py --concurrency=8 --duration=15
```

**Measured 2026-09-08, concurrency 16, 8s per engine, exact (FLAT)
vector search:**

| Engine | QPS | p50 | p95 | p99 | p99.9 | max |
| ------ | --- | --- | --- | --- | ----- | --- |
| Redis | 2,315 | 6.62 ms | 10.52 ms | 12.51 ms | 17.24 ms | 73.67 ms |
| Postgres | 678 | 21.6 ms | 34.88 ms | 53.42 ms | 67.33 ms | 81.59 ms |

**3.41× throughput.** The query vector is embedded once, outside the
timed loop, and reused by every worker — see
[Methodology](docs/METHODOLOGY.md) for why re-embedding per request
would measure Python's GIL more than either datastore.

## Validating the results

```bash
.venv/bin/python src/validate.py
```

Every expected answer — corpus counts, nearest-neighbor results, and
routing decisions — is recomputed **independently**: brute-force cosine
distance in plain NumPy for search, and a from-scratch routing
implementation (not imported from either store) for routing. Two engines
agreeing proves nothing if both are wrong; both are checked against
ground truth computed with no help from either engine.

**Last clean run: 9 checks, 9 passed, 0 failed.**

## Configuration

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `REDIS_URL` | `redis://localhost:6384` | Redis connection |
| `PG_DSN` | `postgresql://demo:demopassword@localhost:5433/semanticdemo` | Postgres connection |
| `PORT` | `8040` | Demo web server |

`REQUEST_EMBED_SAMPLE` in `src/config.py` (default 40,000) controls
how many requests get embedded and searched — the full 300,000-row
table exists in the generated corpus but isn't loaded into either
engine.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| HNSW search returns a wrong category, not just different IDs | Should not happen after the fixes in `redis_store.py`/`seed_redis.py` — if it recurs, see [Methodology](docs/METHODOLOGY.md#two-real-bugs-both-about-hnsw-both-fixed) before assuming it's a data problem. |
| `Port 8040 is in use` | `PORT=8041 .venv/bin/python src/server.py` |
| First embedding call is slow | Expected — downloads and loads the model once; cached after that. |
| Postgres seeding is slow | Building the HNSW index over 40,000 384-dim vectors takes a few seconds; this is normal. |

## Not safe to expose

No authentication, no rate limiting, and the Postgres password is in
`docker-compose.yml`. It's a local demo — keep it on localhost.
