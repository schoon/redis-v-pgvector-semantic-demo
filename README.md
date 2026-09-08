# redis-v-pgvector-semantic-demo

**Vector search**, **hybrid semantic search**, **semantic caching**, and
**semantic intent routing** — all built on **Redis** and **RedisVL** —
over a fictitious bank's customer identity operations: the requests
that keep customer identity records accurate and compliant (address
updates, duplicate TIN/SSN resolution, and related customer-data
maintenance and due-diligence work), a procedure catalog, and a support
SOP knowledge base. Every tab shows per-query latency live, alongside
an explanation of exactly which Redis command or RedisVL feature
produced it.

> **Method and caveats live in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).**
> Read it before presenting — it covers two real HNSW bugs found while
> building this (building the index before the bulk load, and RedisVL's
> default query-time candidate count being too low for this corpus), a
> real timing bug that made routing look far slower than it is (embedding
> the query inside vs. outside the timed call), and what this demo does
> not support (including a newer Redis primitive — Vector Sets —
> considered and deliberately not used here).

No real customer or case data of any kind — every ID, name, and number
in the corpus is synthetic. See
[Methodology](docs/METHODOLOGY.md#no-real-customer-data-anywhere).

## Quick start

```bash
docker compose up -d
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/generate.py
.venv/bin/python src/seed_redis.py
.venv/bin/python src/server.py
```

Then open **<http://localhost:8040>**.

### Before you start

| Need | Why |
| ---- | --- |
| **Docker** with Compose v2 | runs Redis |
| **Python 3.12** | torch/sentence-transformers wheels; newer Pythons may not have them yet |
| **~2 GB free disk** | generated data + the embedding model (downloaded once, cached) |
| Ports **8040**, **6384** free | app, Redis |

First run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)
from Hugging Face and caches it under `~/.cache/huggingface` — no API
key, no network needed on later runs.

### Stopping

```bash
docker compose down
```

No volumes are declared, so this discards the data. Re-run
`seed_redis.py` any time — it flushes Redis first.

## The scenarios

| Scenario | What's running |
| -------- | -------------- |
| **Vector search — exact (FLAT/KNN)** | `FT.SEARCH` against a FLAT index — every candidate compared, no approximation |
| **Vector search — approximate (HNSW)** | `FT.SEARCH` against an HNSW graph index over the same data |
| **Semantic search** | `FT.SEARCH` combining a vector KNN clause with TAG/NUMERIC filters, one query |
| **Semantic caching** | RedisVL `SemanticCache` (built-in extension) — repeat/similar questions hit the cache, skip the search entirely |
| **Semantic routing** | RedisVL `SemanticRouter` (built-in extension); a matched intent immediately reuses the same query embedding to search the right corpus |
| **Concurrent throughput** | `python src/bench.py` — worker threads hitting `FT.SEARCH` concurrently |
| **Architecture** | live schema introspection (`FT.INFO`, `DBSIZE`) |

Every scenario's result is checked against independently-computed
ground truth — see [Validating the results](#validating-the-results).

## Observed on one laptop

**Measured 2026-09-08.** 40,000 embedded requests, 15 procedures,
26 SOP articles, 87 routing reference utterances. Redis 8 in Docker on
a 14-core Apple-silicon MacBook. Median of 3 runs per sample.

| Scenario | Redis |
| -------- | ----- |
| Vector search (exact) | 4.8–6.2 ms |
| Vector search (HNSW) | 1.8–4.1 ms |
| Semantic search — procedures | 1.5–1.9 ms |
| Semantic search — SOP | 1.4–2.6 ms |
| Semantic caching — first call (miss) | 5.5–9.4 ms |
| Semantic caching — repeat calls (avg) | 0.7–2.3 ms |
| Semantic routing | 1.5–2.3 ms |

**Semantic caching's repeat-call numbers are the clearest illustration
of what a cache buys you.** A repeated or paraphrased question skips
`req_flat_idx` entirely after the first time it's asked — the gap
between a first call and a repeat call is the whole exact-search cost
being avoided. Caching the 15-row procedure search instead would barely
be visible against its already-sub-millisecond cost; caching the
40,000-row exact search is where the win actually shows up.

**HNSW recall can read as low (even 0%) on this corpus while still
being correct.** Many requests share literally identical description
text and tie exactly on distance — "the same request IDs as the exact
search" is ambiguous whenever more than 8 rows tie at the 8th-smallest
distance. `validate.py` checks correctness by distance value against a
tie-aware threshold, not raw ID overlap, and passes cleanly even when
the UI's simpler "recall vs exact" percentage looks unimpressive — see
[Methodology](docs/METHODOLOGY.md) for the full explanation.

## Concurrent throughput

```bash
.venv/bin/python src/bench.py
.venv/bin/python src/bench.py --concurrency=8 --duration=15
```

**Measured 2026-09-08, concurrency 16, 8s, exact (FLAT) vector search:**

| Engine | QPS | p50 | p95 | p99 | p99.9 | max |
| ------ | --- | --- | --- | --- | ----- | --- |
| Redis | 2,175 | 6.96 ms | 11.47 ms | 14.1 ms | 27.77 ms | 121.31 ms |

The query vector is embedded once, outside the timed loop, and reused
by every worker — see [Methodology](docs/METHODOLOGY.md) for why
re-embedding per request would measure Python's GIL more than Redis.

## Validating the results

```bash
.venv/bin/python src/validate.py
```

Every expected answer — corpus counts, nearest-neighbor results, and
routing decisions — is recomputed **independently**: brute-force cosine
distance in plain NumPy for search, and a from-scratch routing
implementation (not imported from `redis_store.py`) for routing. Redis
being internally consistent with itself proves nothing on its own;
every check here is against ground truth computed with no help from
RedisVL at all.

**Last clean run: 7 checks, 7 passed, 0 failed.**

## Configuration

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `REDIS_URL` | `redis://localhost:6384` | Redis connection |
| `PORT` | `8040` | Demo web server |

`REQUEST_EMBED_SAMPLE` in `src/config.py` (default 40,000) controls
how many requests get embedded and searched — the full 300,000-row
table exists in the generated corpus but isn't loaded into Redis.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| HNSW search returns a wrong category, not just different IDs | Should not happen after the fixes in `redis_store.py`/`seed_redis.py` — if it recurs, see [Methodology](docs/METHODOLOGY.md) before assuming it's a data problem. |
| `Port 8040 is in use` | `PORT=8041 .venv/bin/python src/server.py` |
| First embedding call is slow | Expected — downloads and loads the model once; cached after that. |

## Not safe to expose

No authentication, no rate limiting. It's a local demo — keep it on
localhost.
