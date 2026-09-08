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
CARDS_PER_CUSTOMER_RANGE = (1, 2)
TRANSACTIONS = 300_000
CARD_PRODUCTS = 15
FAQ_ARTICLES = 26

# Only a sample of transactions get embedded (matches the "don't embed
# everything" convention from the prior demos in this series) — vector
# search over transactions runs against this sample, not the full table.
TRANSACTION_EMBED_SAMPLE = 40_000

DATA_FILES = {
    "customers": os.path.join(DATA_DIR, "customers.jsonl"),
    "cards": os.path.join(DATA_DIR, "cards.jsonl"),
    "transactions": os.path.join(DATA_DIR, "transactions.jsonl"),
    "card_products": os.path.join(DATA_DIR, "card_products.jsonl"),
    "faq_articles": os.path.join(DATA_DIR, "faq_articles.jsonl"),
}
