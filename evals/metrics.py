"""
evals/metrics.py — Pure functions scoring triage-bot's ESI assignments
against clinician-labeled reference vignettes.

No LLM, network, or eval-runner dependency here, on purpose — same
philosophy as agents/assessment.py: the numbers an eval report is built from
have to be independently correct and independently testable, not trusted
just because the report that prints them looks official.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VignetteResult:
    """One vignette's outcome: what the pipeline actually assigned, next to
    what the clinician said it should have been. `assigned_esi` is None when
    the pipeline hit a system error rather than producing a score."""

    vignette_id: str
    reference_esi: int
    assigned_esi: int | None


@dataclass(frozen=True)
class EvalMetrics:
    n: int              # total vignettes attempted
    n_scored: int        # vignettes that actually produced an ESI score
    exact_match_rate: float
    within_one_rate: float
    under_triage_rate: float  # assigned LESS urgent than reference — the dangerous direction
    over_triage_rate: float   # assigned MORE urgent than reference
    system_error_rate: float  # over n, not n_scored — this is the rate of not producing a score at all


def compute_metrics(results: list[VignetteResult]) -> EvalMetrics:
    """
    Agreement rates are computed over successfully-scored vignettes only
    (n_scored), so a pipeline failure doesn't silently count as either a
    match or a miss. system_error_rate is reported separately, over all
    attempted vignettes (n), since that's the rate that actually matters for
    "how often does this fail to produce a usable score at all."
    """
    n = len(results)
    if n == 0:
        return EvalMetrics(n=0, n_scored=0, exact_match_rate=0.0, within_one_rate=0.0,
                            under_triage_rate=0.0, over_triage_rate=0.0, system_error_rate=0.0)

    scored = [r for r in results if r.assigned_esi is not None]
    n_scored = len(scored)
    system_error_rate = (n - n_scored) / n

    if n_scored == 0:
        return EvalMetrics(n=n, n_scored=0, exact_match_rate=0.0, within_one_rate=0.0,
                            under_triage_rate=0.0, over_triage_rate=0.0, system_error_rate=system_error_rate)

    exact = sum(1 for r in scored if r.assigned_esi == r.reference_esi)
    within_one = sum(1 for r in scored if abs(r.assigned_esi - r.reference_esi) <= 1)
    # ESI 1 is the most urgent; a higher number is less urgent. Under-triage
    # is the dangerous direction — the model thought the patient could wait
    # longer than the reference says they actually could.
    under = sum(1 for r in scored if r.assigned_esi > r.reference_esi)
    over = sum(1 for r in scored if r.assigned_esi < r.reference_esi)

    return EvalMetrics(
        n=n,
        n_scored=n_scored,
        exact_match_rate=exact / n_scored,
        within_one_rate=within_one / n_scored,
        under_triage_rate=under / n_scored,
        over_triage_rate=over / n_scored,
        system_error_rate=system_error_rate,
    )
