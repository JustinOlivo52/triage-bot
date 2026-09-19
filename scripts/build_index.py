"""
scripts/build_index.py — Build the clinical reference embedding index.

Run once after editing data/esi_reference.md, then commit the result:

    export VOYAGE_API_KEY=...
    python -m scripts.build_index
    git add data/reference_index.json

The output is committed on purpose. The deployed app never builds an index: it
loads JSON, which means no model download, no vector database, and no write to
an ephemeral filesystem at startup.

Fails loudly on a missing key rather than writing an index without embeddings,
because a silently lexical-only index looks like a successful build and then
quietly degrades retrieval in production.
"""

import logging
import sys

from config import REFERENCE_DOC, REFERENCE_INDEX, VOYAGE_API_KEY
from rag.ingest import build_index


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    log = logging.getLogger("build_index")

    argv = sys.argv[1:] if argv is None else argv
    lexical_only = "--lexical" in argv

    if not lexical_only and not VOYAGE_API_KEY:
        log.error(
            "VOYAGE_API_KEY is not set.\n\n"
            "  export VOYAGE_API_KEY=...\n"
            "  python -m scripts.build_index\n\n"
            "Or build a lexical-only index with no embeddings provider at all:\n\n"
            "  python -m scripts.build_index --lexical\n\n"
            "Lexical retrieval is keyword-based and weaker, but it is still "
            "grounded in the clinical reference and needs no second API key."
        )
        return 1

    try:
        chunks = build_index(REFERENCE_DOC, REFERENCE_INDEX, embed=not lexical_only)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("%s", e)
        return 1

    mode = "lexical-only" if lexical_only else "semantic"
    log.info("Built %d chunks from %s (%s)", len(chunks), REFERENCE_DOC.name, mode)
    log.info("Wrote %s", REFERENCE_INDEX)
    log.info("Commit it: git add %s", REFERENCE_INDEX)

    print("\nSections indexed:")
    for chunk in chunks:
        print(f"  - {chunk.section}  ({len(chunk.text)} chars)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
