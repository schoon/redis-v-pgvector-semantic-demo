# redis-v-pgvector-semantic-demo

Side-by-side **vector search**, **hybrid semantic search**, and
**semantic intent routing** — **Redis (RedisVL)** vs **Postgres
(pgvector)** — over a fictitious credit-card company's transactions,
card products, and support FAQ, with per-query latency shown live.

> **Method and caveats live in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).**
> Read it before presenting — it covers two real HNSW bugs found while
> building this (building the index before the bulk load, and RedisVL's
> default query-time candidate count being too low for this corpus),
> why semantic routing runs the *identical* classification algorithm on
> both engines rather than two similar ones, and what this demo does
> not support.

No real customer or transaction data of any kind — every ID, name, and
number in the corpus is synthetic. See
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
| **Semantic routing** | RedisVL `SemanticRouter` (built-in extension) | hand-rolled SQL replicating the identical algorithm |
| **Concurrent throughput** | `python src/bench.py` | `python src/bench.py` |
| **Architecture** | live schema introspection | live schema introspection |

Both engines embed every query with the exact same local model
(`src/embeddings.py`) — a latency or ranking difference is a
data-access difference, never an embedding difference. Every scenario
reaches the **same result set** on both engines for the same query,
verified against independently-computed ground truth — see
[Validating the results](#validating-the-results).

## Observed on one laptop

**Measured 2026-09-08.** 40,000 embedded transactions, 15 card products,
26 FAQ articles, 87 routing reference utterances. Redis 8 and Postgres
17 + pgvector 0.8.6, both in Docker on a 14-core Apple-silicon MacBook.
Median of 3 runs per sample.

| Scenario | Redis | Postgres | Winner |
| -------- | ----- | -------- | ------ |
| Vector search (exact) | 2.7–4.5 ms | 15.3–18.0 ms | **Redis, ~5–6×** |
| Vector search (HNSW) | 0.9–2.5 ms | 0.4–1.1 ms | **Postgres, ~2×** |
| Semantic search — products | 1.2–2.1 ms | 1.0–1.6 ms | roughly even |
| Semantic search — FAQ | 1.2–1.6 ms | 0.9–2.8 ms | roughly even |
| Semantic routing | 6.9–7.4 ms | 1.4–1.6 ms | **Postgres, ~5×** |

**This is the honest number, not the flattering one, and it's a mixed
result at this scale — deliberately reported that way.** Two scenarios
favor Postgres here:

- **HNSW** — both engines' HNSW result matches their own exact result's
  score exactly (verified — see [Methodology](docs/METHODOLOGY.md)); the
  *specific* transaction IDs returned differ because many transactions
  share identical text and so tie exactly on distance, and each engine
  breaks those ties in its own deterministic order. Postgres happens to
  be a bit faster per query at this row count; that's a real, measured
  result, not an error on either side.
- **Semantic routing** — RedisVL's `SemanticRouter` runs an
  `FT.AGGREGATE` pipeline built to stay fast as a reference set grows
  into the thousands; at 87 reference utterances, a hand-written
  `GROUP BY` in Postgres has less machinery to amortize. Both compute the
  identical classification (verified to 6 decimal places of cosine
  distance — see Methodology) — this is about per-call overhead at small
  scale, not correctness.

**Redis wins decisively where the corpus is largest and where load is
concurrent** — exact vector search over 40,000 rows, and throughput
under concurrency (below). Whether HNSW/routing overhead flips at real
data volume (thousands of routes, millions of vectors) is a real
open question this demo doesn't answer — it's sized for a laptop, not a
production data volume.

## Concurrent throughput

```bash
.venv/bin/python src/bench.py
.venv/bin/python src/bench.py --concurrency=8 --duration=15
```

**Measured 2026-09-08, concurrency 16, 8s per engine, exact (FLAT)
vector search:**

| Engine | QPS | p50 | p95 | p99 | p99.9 | max |
| ------ | --- | --- | --- | --- | ----- | --- |
| Redis | 2,269 | 6.72 ms | 10.89 ms | 12.77 ms | 18.56 ms | 98.4 ms |
| Postgres | 657 | 21.98 ms | 37.94 ms | 55.13 ms | 70.64 ms | 84.5 ms |

**3.45× throughput.** The query vector is embedded once, outside the
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

**Last clean run: 8 checks, 8 passed, 0 failed.**

## Configuration

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `REDIS_URL` | `redis://localhost:6384` | Redis connection |
| `PG_DSN` | `postgresql://demo:demopassword@localhost:5433/semanticdemo` | Postgres connection |
| `PORT` | `8040` | Demo web server |

`TRANSACTION_EMBED_SAMPLE` in `src/config.py` (default 40,000) controls
how many transactions get embedded and searched — the full 300,000-row
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
