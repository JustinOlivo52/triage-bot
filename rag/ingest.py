"""
rag/ingest.py — Document loading, chunking, embedding, and ChromaDB ingestion.
Run directly to force a full re-ingest: python -m rag.ingest
"""

import logging
from functools import lru_cache
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    DATA_DIR,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


# ─── Document Loading ─────────────────────────────────────────────────────────

def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Scan data_dir for PDFs and load every page as a LangChain Document.
    Metadata preserves source filename and page number for traceability.
    """
    documents: list[Document] = []
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDFs found in %s — place clinical guidelines in /data", data_dir)
        return documents

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            page_count = 0
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_num + 1,
                            },
                        )
                    )
                    page_count += 1
            doc.close()
            logger.info("Loaded: %s (%d pages)", pdf_path.name, page_count)
        except Exception as e:
            logger.error("Failed to load %s: %s", pdf_path.name, e)

    logger.info("Total pages loaded: %d", len(documents))
    return documents


# ─── Chunking ─────────────────────────────────────────────────────────────────

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split raw pages into overlapping chunks for embedding.
    Separator order preserves paragraph → sentence → word boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


# ─── Embeddings ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize local HuggingFace embeddings — no API call, no cost.

    Cached: constructing this loads a sentence-transformer model into memory.
    Doing that per query made retrieval latency dominate every triage run.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ─── Vector Store ─────────────────────────────────────────────────────────────

def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and write to persistent ChromaDB on disk."""
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    logger.info("Vector store built — %d chunks persisted to %s", len(chunks), CHROMA_DIR)
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing ChromaDB collection from disk."""
    embeddings = get_embeddings()

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def vector_store_exists() -> bool:
    """Return True if a persisted ChromaDB store is already on disk."""
    return (CHROMA_DIR / "chroma.sqlite3").exists()


# ─── Ingestion Orchestrator ───────────────────────────────────────────────────

def ingest(force: bool = False) -> Chroma | None:
    """
    Full ingestion pipeline: load → chunk → embed → store.

    Skips ingestion and loads from disk if the vector store already exists.
    Pass force=True to re-ingest after adding new documents to /data.

    Returns None when there is nothing to ingest. Building an empty store
    instead would load the embedding model for no reason, which turns a
    missing-guidelines setup into a hard startup failure — the app is supposed
    to run ungrounded in that case, not crash.
    """
    if vector_store_exists() and not force:
        logger.info("Vector store found on disk — loading existing store")
        return load_vector_store()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()

    if not documents:
        logger.warning(
            "No documents found in %s — skipping ingestion. Triage will run "
            "without retrieved clinical context.", DATA_DIR,
        )
        return None

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chunks = split_documents(documents)
    return build_vector_store(chunks)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    logger.info("Starting forced ingest...")
    ingest(force=True)
    logger.info("Ingest complete.")
