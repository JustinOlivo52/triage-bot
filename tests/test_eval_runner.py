"""
tests/test_eval_runner.py — evals/runner.py's aggregation wiring, with
run_triage stubbed at the same boundary the rest of this project's tests
use. This is NOT in the pure-logic CI job: evals/runner.py imports
agents/triage_agent.py, which needs the full LangChain/LangGraph stack to
even import, key or no key.
"""

import evals.runner as runner
from evals.vignettes import Vignette
from models import EscalationAssessment, EscalationLevel, Patient, PatientCard, TriageResult, VitalSigns

_VITALS = VitalSigns(
    heart_rate=80, systolic_bp=120, diastolic_bp=80,
    respiratory_rate=16, spo2=98.0, temperature_c=37.0,
)


def _vignette(vid: str, reference_esi: int) -> Vignette:
    return Vignette(
        id=vid, age=40, weight_kg=70.0, chief_complaint="test complaint",
        vitals=_VITALS, reference_esi=reference_esi, reference_rationale="test",
        reviewed_by_clinician=True,
    )


def _stub_run_triage(monkeypatch, esi_by_id: dict[str, int | None]):
    def _fake(patient: Patient) -> PatientCard:
        esi = esi_by_id[patient.patient_id]
        if esi is None:
            return PatientCard(
                patient=patient, triage_result=None,
                escalation=EscalationAssessment(level=EscalationLevel.IMMEDIATE, system_error="stubbed failure"),
            )
        return PatientCard(
            patient=patient,
            triage_result=TriageResult(esi_score=esi, esi_rationale="r", clinical_reasoning="c"),
            escalation=EscalationAssessment(level=EscalationLevel.NONE),
        )

    monkeypatch.setattr(runner, "run_triage", _fake)


def test_run_eval_matches_assigned_to_reference(monkeypatch):
    vignettes = [_vignette("EVAL-T1", reference_esi=2), _vignette("EVAL-T2", reference_esi=4)]
    _stub_run_triage(monkeypatch, {"EVAL-T1": 2, "EVAL-T2": 5})

    report = runner.run_eval(vignettes)

    assert report["n_vignettes"] == 2
    per = {r["vignette_id"]: r for r in report["per_vignette"]}
    assert per["EVAL-T1"]["assigned_esi"] == 2
    assert per["EVAL-T2"]["assigned_esi"] == 5
    assert report["metrics"]["exact_match_rate"] == 0.5
    assert report["metrics"]["under_triage_rate"] == 0.5


def test_run_eval_handles_a_system_error(monkeypatch):
    vignettes = [_vignette("EVAL-T3", reference_esi=1)]
    _stub_run_triage(monkeypatch, {"EVAL-T3": None})

    report = runner.run_eval(vignettes)

    assert report["metrics"]["n_scored"] == 0
    assert report["metrics"]["system_error_rate"] == 1.0
    assert report["per_vignette"][0]["assigned_esi"] is None


def test_main_refuses_reviewed_only_run_with_no_reviewed_vignettes(monkeypatch, capsys):
    # _vignette() defaults reviewed_by_clinician=True; override it here to
    # simulate the real, all-draft state of the committed vignette set.
    draft = _vignette("X", 3).model_copy(update={"reviewed_by_clinician": False})
    monkeypatch.setattr(runner, "load_vignettes", lambda: [draft])
    monkeypatch.setattr("sys.argv", ["runner"])

    runner.main()

    captured = capsys.readouterr()
    assert "No clinician-reviewed vignettes yet" in captured.out
