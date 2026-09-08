"""
The one place text gets turned into vectors. redis_store.py and
seed_redis.py both import this module rather than rolling their own — so
a given piece of text always produces the exact same vector everywhere
it's used.

sentence-transformers/all-MiniLM-L6-v2, 384 dimensions, runs locally on
CPU — no API key, no network at query time (the model itself is
downloaded once and cached under ~/.cache/huggingface).
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, VECTOR_DIM


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(text):
    """Embed a single string. Returns a plain list[float], length VECTOR_DIM."""
    vec = _model().encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return [float(x) for x in vec]


def embed_many(texts, batch_size=256, show_progress_bar=False):
    """Embed a list of strings. Returns a list[list[float]], same order as input."""
    vecs = _model().encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return [[float(x) for x in row] for row in vecs]


assert VECTOR_DIM == 384, "VECTOR_DIM must match all-MiniLM-L6-v2's real output size"
