import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6384")

PORT = int(os.environ.get("PORT", "8040"))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384

SEED = 20260908

# Fictitious corpus scale — sized to run comfortably on a laptop, not a
# claim about the real scale of any customer's data.
CUSTOMERS = 5_000
REQUESTS = 300_000
PROCEDURES = 15
SOP_ARTICLES = 26

# Only a sample of requests get embedded (matches the "don't embed
# everything" convention from the prior demos in this series) — vector
# search over requests runs against this sample, not the full table.
REQUEST_EMBED_SAMPLE = 40_000

# Seeded data-quality problems for the duplicate-detection demo. TIN
# collisions are baked into customers.jsonl itself (real dirty data,
# loaded into Redis like any other customer); the example counts below
# are held-out test submissions (never loaded into customer_idx) used
# to drive the demo and validate.py.
DUPLICATE_TIN_COUNT = 50
CLEAN_EXAMPLE_COUNT = 10
EXACT_DUP_EXAMPLE_COUNT = 10
NEAR_DUPLICATE_COUNT = 50

# BF.RESERVE sizing for the TIN Bloom filter — capacity comfortably
# above CUSTOMERS (the real number of TINs ever added) so it never
# grows past its designed error rate.
TIN_BLOOM_ERROR_RATE = 0.01
TIN_BLOOM_CAPACITY = 6_000

# Cosine-distance cutoff for flagging a fuzzy name match (city is a
# hard filter, not blended into the embedded text — see
# redis_store.check_duplicate()) as a possible near-duplicate profile.
# Measured against the real generated corpus, not a couple of hand-
# picked examples: at 0.12, this catches 49/50 (98%) of the seeded
# near-duplicate examples' true source in top-3, at a measured 0.56%
# false-positive rate across every real same-last-name-same-city
# namesake pair in the corpus (49 of 8,709). The two distributions
# genuinely overlap in the tail — no threshold gets both to 100%/0% —
# so this is a disclosed tradeoff, not a bug to keep chasing. See
# docs/METHODOLOGY.md for the full measurement.
NEAR_DUPLICATE_DISTANCE_THRESHOLD = 0.12

DATA_FILES = {
    "customers": os.path.join(DATA_DIR, "customers.jsonl"),
    "duplicate_check_examples": os.path.join(DATA_DIR, "duplicate_check_examples.jsonl"),
    "requests": os.path.join(DATA_DIR, "requests.jsonl"),
    "procedures": os.path.join(DATA_DIR, "procedures.jsonl"),
    "sop_articles": os.path.join(DATA_DIR, "sop_articles.jsonl"),
}
