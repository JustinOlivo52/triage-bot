"""
rag/ingest.py — Build-time indexing of the clinical reference.

Runs offline, not at app startup. The resulting index is committed to the repo,
so the deployed app only ever loads JSON — no model download, no vector
database, no write to an ephemeral filesystem.

Rebuild after editing the reference:

    python -m scripts.build_index

Chunking splits on markdown headings rather than a fixed character window. For
a criteria document that is the semantically correct boundary: each chunk is
one self-contained rule, and it carries the heading it came from so the prompt
can cite it.
"""

import json
import logging
import re
from pathlib import Path

from config import MAX_CHUNK_CHARS, REFERENCE_DOC, REFERENCE_INDEX
from models import ReferenceChunk
from rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _split_oversized(text: str, limit: int) -> list[str]:
    """
    Split a section that is too long, preferring paragraph boundaries.

    Only used for sections that exceed the limit; well-sized sections pass
    through whole.
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > limit and current:
            parts.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def chunk_markdown(markdown: str, source: str, limit: int = MAX_CHUNK_CHARS) -> list[ReferenceChunk]:
    """
    Split a markdown document into one chunk per heading section.

    The heading path is retained (e.g. "ESI Levels > ESI 2 — Emergent") so a
    retrieved chunk can be cited precisely rather than by page number.
    """
    headings = list(_HEADING.finditer(markdown))
    if not headings:
        return [
            ReferenceChunk(text=body, source=source, section="(document)")
            for body in _split_oversized(markdown.strip(), limit)
            if body.strip()
        ]

    chunks: list[ReferenceChunk] = []
    path: dict[int, str] = {}

    for i, match in enumerate(headings):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Maintain the heading path: a new heading replaces its own level and
        # clears anything deeper.
        path[level] = title
        for deeper in [lv for lv in path if lv > level]:
            del path[deeper]
        section = " > ".join(path[lv] for lv in sorted(path))

        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        body = markdown[start:end].strip()

        if not body:
            continue  # heading with no content of its own

        for part in _split_oversized(body, limit):
            chunks.append(ReferenceChunk(
                text=f"{section}\n\n{part}",
                source=source,
                section=section,
            ))

    return chunks


# ─── Index Building ───────────────────────────────────────────────────────────

def build_index(
    doc_path: Path = REFERENCE_DOC,
    out_path: Path = REFERENCE_INDEX,
    *,
    api_key: str | None = None,
    embed: bool = True,
) -> list[ReferenceChunk]:
    """
    Chunk the reference and write the index to disk.

    With `embed=True` (the default) an embeddings key is required and the index
    supports semantic search. With `embed=False` the index is written without
    vectors and retrieval runs in lexical mode — worse, but it means the
    project has *no* hard dependency on an embeddings provider.

    Never silently downgrades: asking for embeddings and not getting them
    raises, because an index that looks built but searches lexically would
    quietly degrade production with no visible failure.
    """
    if not doc_path.exists():
        raise FileNotFoundError(
            f"Clinical reference not found at {doc_path}. "
            f"Write it before building the index."
        )

    markdown = doc_path.read_text(encoding="utf-8")
    chunks = chunk_markdown(markdown, source=doc_path.name)
    if not chunks:
        raise ValueError(f"{doc_path} produced no chunks — is it empty?")

    logger.info("Chunked %s into %d sections", doc_path.name, len(chunks))

    if embed:
        vectors = embed_texts([c.text for c in chunks], input_type="document", api_key=api_key)
        if vectors is None:
            raise RuntimeError(
                "Embedding failed. Set VOYAGE_API_KEY to build a semantic index, "
                "or pass embed=False to build a lexical-only index. Refusing to "
                "write an index that claims to be semantic but is not."
            )
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
    else:
        logger.warning(
            "Building a lexical-only index — retrieval will use keyword overlap, "
            "not semantic similarity"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dimensions = len(chunks[0].embedding)
    payload = {
        "source": doc_path.name,
        "chunk_count": len(chunks),
        "dimensions": dimensions,
        "mode": "semantic" if dimensions else "lexical",
        "chunks": [c.model_dump() for c in chunks],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(
        "Wrote %d chunks (%s) to %s",
        len(chunks),
        f"{dimensions} dimensions" if dimensions else "no embeddings",
        out_path,
    )
    return chunks


def load_index(path: Path = REFERENCE_INDEX) -> list[ReferenceChunk]:
    """Load the committed index. Returns an empty list if it is not present."""
    if not path.exists():
        logger.warning("No reference index at %s — triage will run ungrounded", path)
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [ReferenceChunk.model_validate(c) for c in payload["chunks"]]
    except Exception as e:
        logger.error("Could not read reference index %s: %s", path, e)
        return []


def index_exists(path: Path = REFERENCE_INDEX) -> bool:
    return path.exists()
