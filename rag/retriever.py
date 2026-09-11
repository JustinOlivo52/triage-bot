"""
rag/retriever.py — Similarity search and context formatting against ChromaDB.
Used by the triage agent to ground LLM reasoning in clinical guidelines.
"""

import logging

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from config import RETRIEVAL_K
from rag.ingest import load_vector_store, vector_store_exists

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "No clinical reference context available."

# Cached store handle. Loading it constructs the embedding model, which is
# expensive enough that doing it per query dominates triage latency. A failed
# lookup is deliberately not cached, so a store built later is picked up.
_store: Chroma | None = None


# ─── Store Access ─────────────────────────────────────────────────────────────

def get_vector_store() -> Chroma | None:
    """
    Return the ChromaDB vector store if it exists, otherwise None.
    Callers are responsible for handling the None case gracefully.
    """
    global _store

    if _store is not None:
        return _store

    if not vector_store_exists():
        logger.warning("No vector store found — retrieval context will be unavailable")
        return None

    _store = load_vector_store()
    return _store


def reset_vector_store_cache() -> None:
    """Drop the cached store handle. Call after a re-ingest."""
    global _store
    _store = None


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = RETRIEVAL_K) -> list[Document]:
    """
    Run similarity search against ChromaDB and return the top-k Documents.
    Returns an empty list if no vector store exists or the search fails.
    """
    store = get_vector_store()
    if store is None:
        return []

    try:
        results = store.similarity_search(query, k=k)
        logger.info("Retrieved %d chunks for query: '%s'", len(results), query[:80])
        return results
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []


def retrieve_with_scores(query: str, k: int = RETRIEVAL_K) -> list[tuple[Document, float]]:
    """
    Similarity search with relevance scores.
    Returns list of (Document, score) tuples — higher score = more relevant.
    """
    store = get_vector_store()
    if store is None:
        return []

    try:
        results = store.similarity_search_with_relevance_scores(query, k=k)
        logger.info("Retrieved %d scored chunks for query: '%s'", len(results), query[:80])
        return results
    except Exception as e:
        logger.error("Scored retrieval failed: %s", e)
        return []


# ─── Context Formatting ───────────────────────────────────────────────────────

def format_context(documents: list[Document]) -> str:
    """
    Format retrieved Documents into a context block ready for prompt injection.
    Each chunk is labeled with its source and page number.
    """
    if not documents:
        return NO_CONTEXT_MESSAGE

    sections: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "?")
        sections.append(
            f"[Reference {i} — {source}, p.{page}]\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(sections)


# ─── Primary Interface ────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = RETRIEVAL_K) -> tuple[str, bool]:
    """
    Retrieve top-k chunks as a formatted context string.

    Returns (context, found). The boolean matters: on a miss the context string
    is a human-readable placeholder, which is truthy, so callers cannot infer
    grounding from the string alone.
    """
    documents = retrieve(query, k=k)
    return format_context(documents), bool(documents)


def retrieve_triage_context(chief_complaint: str) -> tuple[str, bool]:
    """
    Targeted retrieval for the triage reasoning node. Pulls ESI scoring criteria
    and clinical decision guidelines relevant to the chief complaint.
    """
    query = (
        f"ESI triage scoring criteria for {chief_complaint} "
        f"emergency severity index guidelines"
    )
    return retrieve_context(query)
