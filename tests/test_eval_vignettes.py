"""
tests/test_eval_vignettes.py — evals/vignettes.py's loader, against the real
committed evals/vignettes.json. Pure pydantic — no LLM, no network.

This is a regression test on the vignette file itself: if it's ever hand-
edited into something the schema rejects, this catches it before a real
(costly) eval run does.
"""

from evals.vignettes import Vignette, load_vignettes


def test_loads_the_committed_vignette_set():
    vignettes = load_vignettes()
    assert len(vignettes) >= 20
    assert all(isinstance(v, Vignette) for v in vignettes)


def test_ids_are_unique():
    vignettes = load_vignettes()
    ids = [v.id for v in vignettes]
    assert len(ids) == len(set(ids))


def test_every_esi_level_is_represented():
    vignettes = load_vignettes()
    levels = {v.reference_esi for v in vignettes}
    assert levels == {1, 2, 3, 4, 5}


def test_none_are_marked_reviewed_yet():
    """Documents current reality — flip this vignette-by-vignette as Justin
    actually reviews them, don't just flip the test."""
    vignettes = load_vignettes()
    assert all(v.reviewed_by_clinician is False for v in vignettes)


def test_reference_esi_is_in_valid_range():
    vignettes = load_vignettes()
    assert all(1 <= v.reference_esi <= 5 for v in vignettes)
