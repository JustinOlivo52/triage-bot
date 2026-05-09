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


# ─── Store Access ─────────────────────────────────────────────────────────────

def get_vector_store() -> Chroma | None:
    """
    Return the ChromaDB vector store if it exists, otherwise None.
    Callers are responsible for handling the None case gracefully.
    """
    if not vector_store_exists():
        logger.warning("No vector store found — RAG context will be unavailable")
        return None
    return load_vector_store()


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = RETRIEVAL_K) -> list[Document]:
    """
    Run similarity search against ChromaDB and return the top-k Documents.
    Returns an empty list if no vector store exists.
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
    Format a list of retrieved Documents into a clean context block
    ready to inject into an LLM prompt.
    Each chunk is labeled with its source and page number.
    """
    if not documents:
        return "No clinical reference context available."

    sections: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page   = doc.metadata.get("page", "?")
        sections.append(
            f"[Reference {i} — {source}, p.{page}]\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(sections)


# ─── Primary Interface ────────────────────────────────────────────────────────

def retrieve_context(query: str, k: int = RETRIEVAL_K) -> str:
    """
    Single call used by the triage agent — retrieves top-k chunks
    and returns them as a formatted context string for prompt injection.
    """
    documents = retrieve(query, k=k)
    return format_context(documents)


def retrieve_red_flag_context(patient_age: int, chief_complaint: str) -> str:
    """
    Targeted retrieval for the red flag Layer 3 LLM reasoning node.
    Builds a composite query from patient age and complaint to pull
    the most clinically relevant guidelines for risk assessment.
    """
    age_group = (
        "pediatric patient" if patient_age <= 5
        else "geriatric patient" if patient_age >= 65
        else "adult patient"
    )
    query = (
        f"{age_group} presenting with {chief_complaint} — "
        f"emergency triage risk assessment red flag criteria"
    )
    return retrieve_context(query)


def retrieve_triage_context(chief_complaint: str, esi_hint: str = "") -> str:
    """
    Targeted retrieval for the standard LLM triage reasoning node.
    Pulls ESI scoring criteria and clinical decision guidelines
    relevant to the patient's chief complaint.
    """
    query = (
        f"ESI triage scoring criteria for {chief_complaint} "
        f"{esi_hint} emergency severity index guidelines"
    )
    return retrieve_context(query)
