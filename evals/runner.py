"""
evals/runner.py — Run every vignette through the real triage pipeline and
score the result against its clinician-labeled reference ESI.

Needs a real ANTHROPIC_API_KEY — this measures the actual model, not a stub.
Costs roughly $0.05/vignette at claude-opus-5 pricing (per CLAUDE.md's own
estimate for a live triage run).

    python -m evals.runner                  # clinician-reviewed vignettes only
    python -m evals.runner --include-draft   # + unreviewed drafts, clearly labeled

Refuses to run reviewed-only mode with zero reviewed vignettes rather than
silently reporting an empty result as if it meant something.
"""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from agents.triage_agent import run_triage
from evals.metrics import VignetteResult, compute_metrics
from evals.vignettes import Vignette, load_vignettes
from models import Patient

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def _run_one(vignette: Vignette) -> VignetteResult:
    """
    name="Eval Patient" is a placeholder, not a real field this test
    exercises — the pipeline never sends patient name to the LLM (only
    patient_id), so it has no bearing on the ESI it assigns.
    """
    patient = Patient(
        patient_id=vignette.id,
        name="Eval Patient",
        age=vignette.age,
        weight_kg=vignette.weight_kg,
        chief_complaint=vignette.chief_complaint,
        vitals=vignette.vitals,
    )
    card = run_triage(patient)
    assigned = card.triage_result.esi_score if card.triage_result else None
    return VignetteResult(vignette_id=vignette.id, reference_esi=vignette.reference_esi, assigned_esi=assigned)


def run_eval(vignettes: list[Vignette]) -> dict:
    results = [_run_one(v) for v in vignettes]
    metrics = compute_metrics(results)
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_vignettes": len(vignettes),
        "metrics": asdict(metrics),
        "per_vignette": [asdict(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-draft", action="store_true",
                         help="Also run vignettes not yet reviewed by a clinician")
    args = parser.parse_args()

    all_vignettes = load_vignettes()
    reviewed = [v for v in all_vignettes if v.reviewed_by_clinician]
    draft = [v for v in all_vignettes if not v.reviewed_by_clinician]

    targets = all_vignettes if args.include_draft else reviewed
    if not targets:
        print(
            f"No clinician-reviewed vignettes yet ({len(draft)} draft, unreviewed). "
            "Pass --include-draft to run them anyway — the result is a preview, "
            "not a real accuracy number, and the report says so."
        )
        return

    report = run_eval(targets)
    report["includes_draft"] = args.include_draft
    report["n_reviewed_available"] = len(reviewed)
    report["n_draft_available"] = len(draft)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = report["metrics"]
    label = "reviewed + draft" if args.include_draft else "reviewed only"
    print(f"Ran {m['n_scored']}/{m['n']} vignettes ({label})")
    if args.include_draft and draft:
        print(f"  includes {len(draft)} UNREVIEWED draft vignette(s) — not a certified accuracy number")
    print(f"Exact match:    {m['exact_match_rate']:.1%}")
    print(f"Within one ESI: {m['within_one_rate']:.1%}")
    print(f"Under-triage:   {m['under_triage_rate']:.1%}  (dangerous direction)")
    print(f"Over-triage:    {m['over_triage_rate']:.1%}")
    print(f"System errors:  {m['system_error_rate']:.1%}")
    print(f"Full report: {out_path}")


if __name__ == "__main__":
    main()
