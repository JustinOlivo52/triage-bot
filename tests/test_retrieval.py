"""
Tests for chunking and retrieval.

Like the clinical rule tests, these need no API key and no network. Retrieval
is cosine similarity over a small matrix, so the ranking is exactly testable
with fixture vectors.
"""

import json

import pytest

from models import ReferenceChunk
from rag import retriever
from rag.ingest import build_index, chunk_markdown, load_index

SAMPLE = """\
# Triage Reference

Intro paragraph.

## ESI Levels

Overview of levels.

### ESI 1 — Resuscitation

Requires immediate life-saving intervention.

### ESI 5 — Non-Urgent

No resources needed.

## Vital Signs

Adult ranges only.
"""


@pytest.fixture(autouse=True)
def _clear_index_cache():
    """The index is cached for the process; isolate every test from it."""
    retriever.reset_index_cache()
    yield
    retriever.reset_index_cache()


def chunk(text, embedding=None, section="S", source="ref.md"):
    return ReferenceChunk(
        text=text, source=source, section=section, embedding=embedding or []
    )


# ─── Chunking ─────────────────────────────────────────────────────────────────

class TestChunking:

    def test_splits_on_headings(self):
        chunks = chunk_markdown(SAMPLE, source="ref.md")
        sections = [c.section for c in chunks]
        assert "Triage Reference > ESI Levels > ESI 1 — Resuscitation" in sections

    def test_heading_path_nests_and_resets(self):
        """A new h2 must clear the previous h3, not inherit it."""
        chunks = chunk_markdown(SAMPLE, source="ref.md")
        vitals = [c for c in chunks if c.section.endswith("Vital Signs")]
        assert len(vitals) == 1
        # Must not still carry "ESI 5 — Non-Urgent" from the preceding section.
        assert "ESI" not in vitals[0].section

    def test_heading_with_no_body_is_skipped(self):
        chunks = chunk_markdown("# A\n\n## B\n\n## C\n\nbody\n", source="ref.md")
        assert [c.section for c in chunks] == ["A > C"]

    def test_chunk_text_carries_its_section(self):
        """Retrieved text must be self-describing once it is in a prompt."""
        chunks = chunk_markdown(SAMPLE, source="ref.md")
        assert all(c.text.startswith(c.section) for c in chunks)

    def test_oversized_section_is_split(self):
        long_body = "\n\n".join(["paragraph " * 30] * 10)
        chunks = chunk_markdown(f"# T\n\n{long_body}\n", source="ref.md", limit=500)
        assert len(chunks) > 1

    def test_document_with_no_headings_still_chunks(self):
        chunks = chunk_markdown("Just prose, no headings at all.", source="ref.md")
        assert len(chunks) == 1
        assert chunks[0].section == "(document)"

    def test_citation_format(self):
        c = chunk("body", section="A > B", source="ref.md")
        assert c.citation == "ref.md § A > B"


# ─── Index Building ───────────────────────────────────────────────────────────

class TestIndexBuilding:

    def test_lexical_index_round_trips(self, tmp_path):
        doc = tmp_path / "ref.md"
        doc.write_text(SAMPLE, encoding="utf-8")
        out = tmp_path / "index.json"

        built = build_index(doc, out, embed=False)
        loaded = load_index(out)

        assert len(loaded) == len(built)
        assert [c.section for c in loaded] == [c.section for c in built]
        assert json.loads(out.read_text())["mode"] == "lexical"

    def test_embed_true_without_a_key_raises(self, tmp_path, monkeypatch):
        """
        Must not silently write a lexical index when asked for a semantic one —
        that would look like a successful build and quietly degrade production.
        """
        monkeypatch.setattr("rag.ingest.embed_texts", lambda *a, **k: None)
        doc = tmp_path / "ref.md"
        doc.write_text(SAMPLE, encoding="utf-8")

        with pytest.raises(RuntimeError, match="Refusing to write"):
            build_index(doc, tmp_path / "index.json", embed=True)
        assert not (tmp_path / "index.json").exists()

    def test_missing_document_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_index(tmp_path / "nope.md", tmp_path / "index.json", embed=False)

    def test_empty_document_raises(self, tmp_path):
        doc = tmp_path / "ref.md"
        doc.write_text("   \n\n  ", encoding="utf-8")
        with pytest.raises(ValueError):
            build_index(doc, tmp_path / "index.json", embed=False)

    def test_missing_index_loads_as_empty(self, tmp_path):
        assert load_index(tmp_path / "absent.json") == []

    def test_corrupt_index_loads_as_empty_rather_than_raising(self, tmp_path):
        bad = tmp_path / "index.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_index(bad) == []


# ─── Semantic Search ──────────────────────────────────────────────────────────

class TestSemanticSearch:
    """Fixture vectors make cosine ranking exactly predictable."""

    CHUNKS = [
        chunk("cardiac", embedding=[1.0, 0.0, 0.0], section="Cardiac"),
        chunk("respiratory", embedding=[0.0, 1.0, 0.0], section="Respiratory"),
        chunk("neuro", embedding=[0.0, 0.0, 1.0], section="Neuro"),
    ]

    def _setup(self, monkeypatch, query_vector):
        monkeypatch.setattr(retriever, "load_index", lambda *a, **k: list(self.CHUNKS))
        monkeypatch.setattr(retriever, "embeddings_available", lambda: True)
        monkeypatch.setattr(retriever, "embed_query", lambda q: query_vector)
        retriever.reset_index_cache()

    def test_ranks_by_cosine_similarity(self, monkeypatch):
        self._setup(monkeypatch, [0.0, 1.0, 0.0])
        assert [c.section for c in retriever.retrieve("q", k=1)] == ["Respiratory"]

    def test_magnitude_does_not_affect_ranking(self, monkeypatch):
        """Cosine is scale-invariant; a longer vector must not win on length."""
        self._setup(monkeypatch, [0.0, 99.0, 0.0])
        assert retriever.retrieve("q", k=1)[0].section == "Respiratory"

    def test_k_limits_results(self, monkeypatch):
        self._setup(monkeypatch, [1.0, 0.0, 0.0])
        assert len(retriever.retrieve("q", k=2)) == 2

    def test_mode_reports_semantic(self, monkeypatch):
        self._setup(monkeypatch, [1.0, 0.0, 0.0])
        assert retriever.retrieval_mode() == "semantic"

    def test_dimension_mismatch_falls_back_instead_of_crashing(self, monkeypatch):
        """An index built with a different model must not raise at query time."""
        self._setup(monkeypatch, [1.0, 0.0])  # wrong width
        results = retriever.retrieve("cardiac", k=2)
        assert isinstance(results, list)  # fell through to lexical

    def test_failed_query_embedding_falls_back_to_lexical(self, monkeypatch):
        monkeypatch.setattr(retriever, "load_index", lambda *a, **k: list(self.CHUNKS))
        monkeypatch.setattr(retriever, "embeddings_available", lambda: True)
        monkeypatch.setattr(retriever, "embed_query", lambda q: None)
        retriever.reset_index_cache()

        assert retriever.retrieve("cardiac", k=1)[0].section == "Cardiac"


# ─── Lexical Fallback ─────────────────────────────────────────────────────────

class TestLexicalSearch:

    CHUNKS = [
        chunk("chest pain radiating to the jaw", section="Cardiac"),
        chunk("ankle sprain and minor limb injury", section="Ortho"),
        chunk("fever tachycardia suspected sepsis", section="Infectious"),
    ]

    def _setup(self, monkeypatch):
        monkeypatch.setattr(retriever, "load_index", lambda *a, **k: list(self.CHUNKS))
        monkeypatch.setattr(retriever, "embeddings_available", lambda: False)
        retriever.reset_index_cache()

    def test_matches_on_token_overlap(self, monkeypatch):
        self._setup(monkeypatch)
        assert retriever.retrieve("chest pain", k=1)[0].section == "Cardiac"

    def test_mode_reports_lexical_without_a_key(self, monkeypatch):
        self._setup(monkeypatch)
        assert retriever.retrieval_mode() == "lexical"

    def test_query_of_only_stopwords_returns_nothing(self, monkeypatch):
        self._setup(monkeypatch)
        assert retriever.retrieve("the and of to", k=3) == []

    def test_no_overlap_returns_nothing(self, monkeypatch):
        self._setup(monkeypatch)
        assert retriever.retrieve("xylophone bassoon", k=3) == []


# ─── Degradation ──────────────────────────────────────────────────────────────

class TestDegradation:
    """Retrieval is an enhancement to triage, never a precondition for it."""

    def test_missing_index_yields_no_context_and_found_false(self, monkeypatch):
        monkeypatch.setattr(retriever, "load_index", lambda *a, **k: [])
        retriever.reset_index_cache()

        context, found = retriever.retrieve_context("chest pain")
        assert found is False
        assert context == retriever.NO_CONTEXT_MESSAGE

    def test_missing_index_reports_mode_none(self, monkeypatch):
        monkeypatch.setattr(retriever, "load_index", lambda *a, **k: [])
        retriever.reset_index_cache()
        assert retriever.retrieval_mode() == "none"

    def test_found_flag_is_not_inferable_from_the_string(self, monkeypatch):
        """
        Regression: the miss placeholder is truthy, which is why callers get an
        explicit boolean rather than testing the string.
        """
        monkeypatch.setattr(retriever, "load_index", lambda *a, **k: [])
        retriever.reset_index_cache()
        context, found = retriever.retrieve_context("anything")
        assert bool(context) is True and found is False

    def test_hit_reports_found_true(self, monkeypatch):
        monkeypatch.setattr(
            retriever, "load_index",
            lambda *a, **k: [chunk("chest pain criteria", section="Cardiac")],
        )
        monkeypatch.setattr(retriever, "embeddings_available", lambda: False)
        retriever.reset_index_cache()

        context, found = retriever.retrieve_context("chest pain")
        assert found is True
        assert "Cardiac" in context


# ─── Context Formatting ───────────────────────────────────────────────────────

class TestFormatting:

    def test_includes_numbered_citations(self):
        out = retriever.format_context([
            chunk("alpha", section="A", source="ref.md"),
            chunk("beta", section="B", source="ref.md"),
        ])
        assert "[Reference 1 — ref.md § A]" in out
        assert "[Reference 2 — ref.md § B]" in out

    def test_empty_gives_the_placeholder(self):
        assert retriever.format_context([]) == retriever.NO_CONTEXT_MESSAGE
