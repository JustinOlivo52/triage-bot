"""
tests/test_eval_metrics.py — evals/metrics.py's pure aggregation functions.
No LLM, no network — part of the "pure-logic" CI job.
"""

from evals.metrics import VignetteResult, compute_metrics


def _r(vignette_id: str, reference_esi: int, assigned_esi: int | None) -> VignetteResult:
    return VignetteResult(vignette_id=vignette_id, reference_esi=reference_esi, assigned_esi=assigned_esi)


def test_empty_results():
    m = compute_metrics([])
    assert m.n == 0
    assert m.n_scored == 0
    assert m.exact_match_rate == 0.0
    assert m.system_error_rate == 0.0


def test_all_exact_matches():
    results = [_r("A", 1, 1), _r("B", 3, 3), _r("C", 5, 5)]
    m = compute_metrics(results)
    assert m.n == 3
    assert m.n_scored == 3
    assert m.exact_match_rate == 1.0
    assert m.within_one_rate == 1.0
    assert m.under_triage_rate == 0.0
    assert m.over_triage_rate == 0.0
    assert m.system_error_rate == 0.0


def test_under_triage_direction():
    """ESI 1 is most urgent; a higher assigned number than reference means
    the model thought the patient could wait longer — under-triage."""
    results = [_r("A", 1, 3)]
    m = compute_metrics(results)
    assert m.under_triage_rate == 1.0
    assert m.over_triage_rate == 0.0
    assert m.exact_match_rate == 0.0
    assert m.within_one_rate == 0.0


def test_over_triage_direction():
    results = [_r("A", 4, 2)]
    m = compute_metrics(results)
    assert m.over_triage_rate == 1.0
    assert m.under_triage_rate == 0.0


def test_within_one_but_not_exact():
    results = [_r("A", 2, 3)]
    m = compute_metrics(results)
    assert m.exact_match_rate == 0.0
    assert m.within_one_rate == 1.0


def test_off_by_more_than_one_is_neither():
    results = [_r("A", 1, 4)]
    m = compute_metrics(results)
    assert m.exact_match_rate == 0.0
    assert m.within_one_rate == 0.0
    assert m.under_triage_rate == 1.0


def test_system_errors_excluded_from_agreement_rates_but_counted_separately():
    results = [_r("A", 1, 1), _r("B", 2, None)]
    m = compute_metrics(results)
    assert m.n == 2
    assert m.n_scored == 1
    assert m.exact_match_rate == 1.0  # computed over n_scored, not n
    assert m.system_error_rate == 0.5  # computed over n


def test_all_system_errors():
    results = [_r("A", 1, None), _r("B", 2, None)]
    m = compute_metrics(results)
    assert m.n == 2
    assert m.n_scored == 0
    assert m.system_error_rate == 1.0
    assert m.exact_match_rate == 0.0
