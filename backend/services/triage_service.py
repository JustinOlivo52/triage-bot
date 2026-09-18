"""
backend/services/triage_service.py — The FHIR persistence boundary.

This is the one place that knows both languages: the pipeline's native
`Patient`/`PatientCard` (models.py) and the FHIR-shaped rows (backend/models/).
`agents/triage_agent.py` is called completely unchanged — it has no idea FHIR
exists, and it shouldn't need to.

Simplification stated once, here: every check-in creates a fresh `Patient`
row and a fresh `patient_identifier` (PT-0001, PT-0002, ...), the same way
the V1 file store did — it does not merge a returning patient's visits onto
one Patient row across encounters. `is_returning` is still computed (matched
by name, case-insensitive) and carried onto the domain Patient for display,
same as V1. Consolidating one person's repeat visits under a single FHIR
Patient with multiple Encounters is a reasonable real-EHR refinement, but
it's not required for this vertical slice and adds real complexity (identity
matching, merge conflicts) that belongs in its own pass, not this one.

`patient_identifier` generation (count existing rows + 1) is not
concurrency-safe — two simultaneous check-ins could race for the same
number. Acceptable for a portfolio project's demo traffic; a production
system would use a DB sequence or a unique-constraint-and-retry loop.
"""

from datetime import date
from typing import NamedTuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from agents.triage_agent import run_triage
from backend.models.audit_log import AuditAction, AuditLog
from backend.models.encounter import Encounter
from backend.models.observation import VITAL_LOINC_CODES
from backend.models.observation import Observation as ObservationRow
from backend.models.patient import Patient as PatientRow
from backend.models.risk_assessment import QUALITATIVE_RISK_BY_ESCALATION, RiskAssessment
from backend.models.user import User
from backend.schemas.encounter import CheckInRequest
from config import PATIENT_ID_PREFIX, TRIAGE_MODEL
from models import Patient as DomainPatient, PatientCard


class CheckInResult(NamedTuple):
    encounter: Encounter
    card: PatientCard


def _estimate_birth_date(age: int) -> date:
    """Today's date minus `age` years — see backend/models/patient.py for why."""
    today = date.today()
    try:
        return today.replace(year=today.year - age)
    except ValueError:
        # today is Feb 29 and (today.year - age) is not a leap year.
        return today.replace(month=2, day=28, year=today.year - age)


def _next_patient_identifier(db: Session) -> str:
    count = db.query(func.count(PatientRow.id)).scalar() or 0
    return f"{PATIENT_ID_PREFIX}-{count + 1:04d}"


def _is_returning(db: Session, full_name: str) -> bool:
    return (
        db.query(PatientRow)
        .filter(func.lower(PatientRow.full_name) == full_name.strip().lower())
        .first()
        is not None
    )


def _write_vital_observations(
    db: Session, patient_row: PatientRow, encounter: Encounter, payload: CheckInRequest
) -> None:
    """One Observation row per vital, including weight — see module docstring
    on backend/models/observation.py for why this isn't one JSON blob."""
    values = {**payload.vitals.model_dump(), "weight_kg": payload.weight_kg}
    for field_name, value in values.items():
        loinc = VITAL_LOINC_CODES[field_name]
        db.add(ObservationRow(
            patient_id=patient_row.id,
            encounter_id=encounter.id,
            code=loinc.code,
            code_display=loinc.display,
            value=float(value),
            unit=loinc.unit,
        ))


def check_in_and_triage(db: Session, payload: CheckInRequest, actor: User) -> CheckInResult:
    """
    Check a patient in, run the existing triage pipeline unchanged, and
    persist the result as FHIR-shaped rows: Patient, Encounter, one
    Observation per vital, and (when scoring succeeded) a RiskAssessment.

    A system error is not persisted as a RiskAssessment with a fabricated
    outcome — see risk_assessment.py's docstring. It's still audited, so the
    failure itself is part of the accountable record.
    """
    is_returning = _is_returning(db, payload.full_name)

    patient_row = PatientRow(
        patient_identifier=_next_patient_identifier(db),
        full_name=payload.full_name,
        birth_date=_estimate_birth_date(payload.age),
        birth_date_is_estimated=True,
    )
    db.add(patient_row)
    db.flush()  # populate patient_row.id for the FK columns below

    encounter = Encounter(
        patient_id=patient_row.id,
        reason_text=payload.chief_complaint,
        age_at_encounter=payload.age,
    )
    db.add(encounter)
    db.flush()

    check_in_audit_row = AuditLog(
        actor_user_id=actor.id,
        action=AuditAction.PATIENT_CHECK_IN,
        resource_type="encounter",
        resource_id=encounter.id,
    )
    check_in_audit_row.metadata_dict = {"patient_identifier": patient_row.patient_identifier}
    db.add(check_in_audit_row)

    _write_vital_observations(db, patient_row, encounter, payload)

    domain_patient = DomainPatient(
        patient_id=patient_row.patient_identifier,
        name=payload.full_name,
        age=payload.age,
        weight_kg=payload.weight_kg,
        chief_complaint=payload.chief_complaint,
        vitals=payload.vitals,
        is_returning=is_returning,
    )

    # The pipeline itself — unchanged, no FHIR awareness.
    card: PatientCard = run_triage(domain_patient)

    if card.triage_result is not None:
        risk_row = RiskAssessment(
            patient_id=patient_row.id,
            encounter_id=encounter.id,
            prediction_outcome=card.display_esi,
            qualitative_risk=QUALITATIVE_RISK_BY_ESCALATION[card.escalation.level.value],
            basis="\n".join(f.detail for f in card.escalation.findings) or "No findings.",
            rationale=card.triage_result.clinical_reasoning,
            performer_model=TRIAGE_MODEL,
        )
        db.add(risk_row)
        audit_row = AuditLog(
            actor_user_id=actor.id,
            action=AuditAction.TRIAGE_RUN,
            resource_type="encounter",
            resource_id=encounter.id,
        )
        audit_row.metadata_dict = {
            "esi_score": card.triage_result.esi_score,
            "escalation": card.escalation.level.value,
        }
    else:
        audit_row = AuditLog(
            actor_user_id=actor.id,
            action=AuditAction.TRIAGE_SYSTEM_ERROR,
            resource_type="encounter",
            resource_id=encounter.id,
        )
        audit_row.metadata_dict = {"error": card.escalation.system_error or "unknown"}
    db.add(audit_row)

    db.commit()
    db.refresh(encounter)

    return CheckInResult(encounter=encounter, card=card)
