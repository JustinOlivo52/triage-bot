"""
rag/embeddings.py — Voyage AI embedding client.

One HTTP call, no local model. This is the whole reason the project does not
carry torch: embedding roughly forty paragraphs of reference material does not
justify shipping a deep-learning framework.

Every function here returns None on failure rather than raising. Retrieval is
an enhancement to triage, not a precondition for it, so a missing key or a
failed request must degrade the result rather than take down the pipeline.
"""

import logging

import requests

from config import (
    EMBEDDING_ENDPOINT,
    EMBEDDING_MODEL,
    EMBEDDING_TIMEOUT_SECONDS,
    VOYAGE_API_KEY,
)

logger = logging.getLogger(__name__)

# Voyage accepts at most 128 inputs per request.
_MAX_BATCH = 128


def embeddings_available() -> bool:
    """True if an embeddings key is configured."""
    return bool(VOYAGE_API_KEY)


def embed_texts(
    texts: list[str],
    input_type: str = "document",
    *,
    api_key: str | None = None,
) -> list[list[float]] | None:
    """
    Embed a batch of texts.

    `input_type` is "document" when building the index and "query" at search
    time — Voyage embeds the two asymmetrically, and using the wrong one
    degrades retrieval quality without producing any visible error.

    Returns None if no key is configured or the request fails.
    """
    if not texts:
        return []

    key = api_key or VOYAGE_API_KEY
    if not key:
        logger.info("No embeddings key configured — semantic search unavailable")
        return None

    vectors: list[list[float]] = []

    for start in range(0, len(texts), _MAX_BATCH):
        batch = texts[start:start + _MAX_BATCH]
        try:
            response = requests.post(
                EMBEDDING_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": batch,
                    "model": EMBEDDING_MODEL,
                    "input_type": input_type,
                },
                timeout=EMBEDDING_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as e:
            logger.error("Embedding request failed: %s", e)
            return None

        try:
            # Results are returned with an explicit index; sort rather than
            # trusting positional order.
            rows = sorted(payload["data"], key=lambda r: r.get("index", 0))
            vectors.extend(row["embedding"] for row in rows)
        except (KeyError, TypeError) as e:
            logger.error("Unexpected embedding response shape: %s", e)
            return None

    if len(vectors) != len(texts):
        logger.error(
            "Embedding count mismatch: asked for %d, received %d",
            len(texts), len(vectors),
        )
        return None

    return vectors


def embed_query(text: str) -> list[float] | None:
    """Embed a single search query. Returns None if unavailable."""
    result = embed_texts([text], input_type="query")
    return result[0] if result else None
