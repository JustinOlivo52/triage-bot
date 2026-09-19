"""
rag/retriever.py — Runtime retrieval over the committed reference index.

Search is cosine similarity over a few dozen precomputed vectors, which is a
numpy dot product, not a database. At this corpus size a vector store would be
pure overhead.

Retrieval degrades in two steps rather than failing:

    embeddings key present  → semantic search
    embeddings key absent   → lexical overlap over the same chunks
    index absent            → no context; triage proceeds ungrounded

So the app is fully functional with one secret (ANTHROPIC_API_KEY); the
embeddings key improves retrieval rather than enabling the app.
"""

import logging
import re
from functools import lru_cache

import numpy as np

from config import RETRIEVAL_K
from models import ReferenceChunk
from rag.embeddings import embed_query, embeddings_available
from rag.ingest import index_exists, load_index

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "No clinical reference context available."

_TOKEN = re.compile(r"[a-z0-9]+")

# Words too common in a clinical corpus to carry signal.
_STOPWORDS = frozenset("""
a an the and or of to in for with without on at by is are was were be been
patient patients presenting presents history the this that these those
""".split())


# ─── Index Access ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _index() -> tuple[tuple[ReferenceChunk, ...], np.ndarray | None]:
    """
    Load the reference index once and precompute its normalised matrix.

    Cached because the index is immutable at runtime — it is a committed file,
    never rebuilt by the running app.
    """
    chunks = load_index()
    if not chunks:
        return (), None

    vectors = [c.embedding for c in chunks if c.embedding]
    if len(vectors) != len(chunks):
        logger.warning("Index has chunks without embeddings — lexical search only")
        return tuple(chunks), None

    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against a zero vector making the whole row NaN.
    matrix = matrix / np.where(norms == 0, 1.0, norms)

    logger.info("Loaded reference index: %d chunks, %d dimensions", *matrix.shape)
    return tuple(chunks), matrix


def reset_index_cache() -> None:
    """Drop the cached index. Call after rebuilding it."""
    _index.cache_clear()


def retrieval_mode() -> str:
    """Which retrieval path is active: 'semantic', 'lexical', or 'none'."""
    chunks, matrix = _index()
    if not chunks:
        return "none"
    if matrix is not None and embeddings_available():
        return "semantic"
    return "lexical"


# ─── Search ───────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def _lexical_search(query: str, chunks: tuple[ReferenceChunk, ...], k: int) -> list[ReferenceChunk]:
    """
    Token-overlap ranking, used when no embeddings key is configured.

    Deliberately simple. It is a fallback that keeps the app useful without a
    second API key, not a competitor to semantic search.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, int, ReferenceChunk]] = []
    for i, chunk in enumerate(chunks):
        chunk_tokens = _tokens(chunk.text)
        if not chunk_tokens:
            continue
        overlap = len(query_tokens & chunk_tokens)
        if overlap:
            # Normalise by chunk size so long sections do not dominate.
            scored.append((overlap / (len(chunk_tokens) ** 0.5), i, chunk))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [chunk for _, _, chunk in scored[:k]]


def _semantic_search(
    query: str,
    chunks: tuple[ReferenceChunk, ...],
    matrix: np.ndarray,
    k: int,
) -> list[ReferenceChunk] | None:
    """Cosine similarity over the precomputed matrix. None if embedding fails."""
    vector = embed_query(query)
    if vector is None:
        return None

    q = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm == 0:
        return None
    q = q / norm

    if q.shape[0] != matrix.shape[1]:
        logger.error(
            "Query dimension %d does not match index dimension %d — "
            "the index was built with a different embedding model",
            q.shape[0], matrix.shape[1],
        )
        return None

    scores = matrix @ q
    top = np.argsort(-scores)[:k]
    return [chunks[i] for i in top]


def retrieve(query: str, k: int = RETRIEVAL_K) -> list[ReferenceChunk]:
    """
    Return the top-k reference chunks for a query.

    Falls back from semantic to lexical automatically, so a missing or failing
    embeddings key costs retrieval quality rather than retrieval itself.
    """
    chunks, matrix = _index()
    if not chunks:
        return []

    if matrix is not None and embeddings_available():
        results = _semantic_search(query, chunks, matrix, k)
        if results is not None:
            logger.info("Semantic retrieval returned %d chunks", len(results))
            return results
        logger.warning("Semantic search unavailable — falling back to lexical")

    results = _lexical_search(query, chunks, k)
    logger.info("Lexical retrieval returned %d chunks", len(results))
    return results


# ─── Context Formatting ───────────────────────────────────────────────────────

def format_context(chunks: list[ReferenceChunk]) -> str:
    """Format retrieved chunks into a citable context block for the prompt."""
    if not chunks:
        return NO_CONTEXT_MESSAGE

    return "\n\n---\n\n".join(
        f"[Reference {i} — {chunk.citation}]\n{chunk.text.strip()}"
        for i, chunk in enumerate(chunks, start=1)
    )


# ─── Primary Interface ────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = RETRIEVAL_K) -> tuple[str, bool]:
    """
    Retrieve top-k chunks as a formatted context string.

    Returns (context, found). The boolean matters: on a miss the context string
    is a human-readable placeholder, which is truthy, so callers cannot infer
    grounding from the string alone.
    """
    chunks = retrieve(query, k=k)
    return format_context(chunks), bool(chunks)


def retrieve_triage_context(chief_complaint: str) -> tuple[str, bool]:
    """
    Targeted retrieval for the triage reasoning node. Pulls ESI scoring criteria
    relevant to the chief complaint.
    """
    query = (
        f"ESI triage scoring criteria for {chief_complaint} "
        f"emergency severity index acuity assignment"
    )
    return retrieve_context(query)
