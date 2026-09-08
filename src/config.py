import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6384")

PG_DSN = os.environ.get(
    "PG_DSN", "postgresql://demo:demopassword@localhost:5433/semanticdemo"
)

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

DATA_FILES = {
    "customers": os.path.join(DATA_DIR, "customers.jsonl"),
    "requests": os.path.join(DATA_DIR, "requests.jsonl"),
    "procedures": os.path.join(DATA_DIR, "procedures.jsonl"),
    "sop_articles": os.path.join(DATA_DIR, "sop_articles.jsonl"),
}
