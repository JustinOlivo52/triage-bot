"""
backend/services/triage_service.py — The FHIR persistence boundary.

This is the one place that knows both languages: the pipeline's native
`Patient`/`PatientCard` (models.py) and the FHIR-shaped rows (backend/models/).
`agents/triage_agent.py` is called completely unchanged — it has no idea FHIR
exists, and it shouldn't need to.

A returning patient (matched by full name, case-insensitive) reuses their
existing `Patient` row and `patient_identifier` rather than getting a new
one each visit — one person, one chart number, multiple `Encounter`s, the
same way a real EHR works. `Patient.birth_date` is estimated once, at the
first visit, and never overwritten by a later visit's reported age — a
person's actual birth date doesn't change. `Encounter.age_at_encounter`
carries the age reported *at that specific visit* instead, which is exactly
why that field exists separately from `Patient.birth_date` (see
backend/models/encounter.py's docstring — it anticipated this before it was
built). Identity matching is name-only, no fuzzier than that: two different
people who happen to share a name would incorrectly merge, and the same
person spelled two different ways would incorrectly stay separate. Real
identity resolution (DOB + name, or a patient-supplied identifier) is a
further refinement, not done here.

`patient_identifier` generation for a genuinely new patient (count existing
rows + 1) is not concurrency-safe — two simultaneous first-time check-ins
could race for the same number. Acceptable for a portfolio project's demo
traffic; a production system would use a DB sequence or a
unique-constraint-and-retry loop.

`Encounter.card_json` (see backend/models/encounter.py) is what every read
below actually returns — the FHIR rows are written for anyone querying the
structured/coded representation, but reconstructing the richer PatientCard
from them would mean re-parsing free text. The cache is the fast path.
"""

from datetime import date, datetime, timezone
from typing import NamedTuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from agents.triage_agent import run_triage
from backend.models.audit_log import AuditAction, AuditLog
from backend.models.encounter import Encounter, EncounterStatus
from backend.models.observation import VITAL_LOINC_CODES
from backend.models.observation import Observation as ObservationRow
from backend.models.patient import Patient as PatientRow
from backend.models.risk_assessment import QUALITATIVE_RISK_BY_ESCALATION, RiskAssessment
from backend.models.user import User
from backend.schemas.encounter import CheckInRequest
from config import PATIENT_ID_PREFIX, TRIAGE_MODEL
from models import Patient as DomainPatient, PatientCard, TriageStatus, VitalSigns


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


def _find_existing_patient(db: Session, full_name: str) -> PatientRow | None:
    """Match by name, case-insensitive — the only identity signal check-in
    actually collects. See the module docstring for what this does and
    doesn't handle correctly."""
    return (
        db.query(PatientRow)
        .filter(func.lower(PatientRow.full_name) == full_name.strip().lower())
        .first()
    )


def _write_vital_observations(
    db: Session, patient_row: PatientRow, encounter: Encounter, vitals: VitalSigns, weight_kg: float
) -> None:
    """One Observation row per vital, including weight — see module docstring
    on backend/models/observation.py for why this isn't one JSON blob."""
    values = {**vitals.model_dump(), "weight_kg": weight_kg}
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


def _load_card(encounter: Encounter) -> PatientCard:
    return PatientCard.model_validate_json(encounter.card_json)


# ─── Check-In + Triage ─────────────────────────────────────────────────────────

def check_in_and_triage(db: Session, payload: CheckInRequest, actor: User) -> CheckInResult:
    """
    Check a patient in, run the existing triage pipeline unchanged, and
    persist the result as FHIR-shaped rows: Patient, Encounter, one
    Observation per vital, and (when scoring succeeded) a RiskAssessment.

    A system error is not persisted as a RiskAssessment with a fabricated
    outcome — see risk_assessment.py's docstring. It's still audited, so the
    failure itself is part of the accountable record.
    """
    existing_patient = _find_existing_patient(db, payload.full_name)
    is_returning = existing_patient is not None

    if existing_patient is not None:
        patient_row = existing_patient
    else:
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

    _write_vital_observations(db, patient_row, encounter, payload.vitals, payload.weight_kg)

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

    encounter.card_json = card.model_dump_json()
    db.commit()
    db.refresh(encounter)

    return CheckInResult(encounter=encounter, card=card)


def persist_seeded_card(db: Session, card: PatientCard, actor: User) -> Encounter:
    """
    Persist a precomputed PatientCard (data/seed_cohort.json) as FHIR rows,
    without re-running the pipeline — the whole point of a seeded cohort is
    that browsing it costs nothing. Reuses the card's own patient_id as the
    patient_identifier, so seeded IDs (PT-0001, ...) stay stable across
    restarts instead of being renumbered by _next_patient_identifier.
    """
    patient = card.patient
    patient_row = PatientRow(
        patient_identifier=patient.patient_id,
        full_name=patient.name,
        birth_date=_estimate_birth_date(patient.age),
        birth_date_is_estimated=True,
    )
    db.add(patient_row)
    db.flush()

    encounter = Encounter(
        patient_id=patient_row.id,
        reason_text=patient.chief_complaint,
        age_at_encounter=patient.age,
        card_json=card.model_dump_json(),
    )
    db.add(encounter)
    db.flush()

    _write_vital_observations(db, patient_row, encounter, patient.vitals, patient.weight_kg)

    if card.triage_result is not None:
        db.add(RiskAssessment(
            patient_id=patient_row.id,
            encounter_id=encounter.id,
            prediction_outcome=card.display_esi,
            qualitative_risk=QUALITATIVE_RISK_BY_ESCALATION[card.escalation.level.value],
            basis="\n".join(f.detail for f in card.escalation.findings) or "No findings.",
            rationale=card.triage_result.clinical_reasoning,
            performer_model=TRIAGE_MODEL,
        ))

    audit_row = AuditLog(
        actor_user_id=actor.id,
        action=AuditAction.PATIENT_CHECK_IN,
        resource_type="encounter",
        resource_id=encounter.id,
    )
    audit_row.metadata_dict = {"patient_identifier": patient_row.patient_identifier, "seeded": True}
    db.add(audit_row)

    return encounter


# ─── Reads ───────────────────────────────────────────────────────────────────

def list_encounters(db: Session, active_only: bool = False) -> list[CheckInResult]:
    """Every encounter, newest check-in first. `active_only` drops anyone a
    clinician has already resolved — same distinction the queue/all-patients
    tabs draw in the UI."""
    rows = (
        db.query(Encounter)
        .filter(Encounter.card_json.isnot(None))
        .order_by(Encounter.period_start.desc())
        .all()
    )
    results = [CheckInResult(encounter=e, card=_load_card(e)) for e in rows]
    if active_only:
        results = [r for r in results if r.card.patient.status is not TriageStatus.RESOLVED]
    return results


def get_encounter_by_patient_identifier(db: Session, patient_identifier: str) -> CheckInResult | None:
    patient_row = (
        db.query(PatientRow)
        .filter(func.upper(PatientRow.patient_identifier) == patient_identifier.strip().upper())
        .first()
    )
    if patient_row is None:
        return None
    encounter = (
        db.query(Encounter)
        .filter(Encounter.patient_id == patient_row.id, Encounter.card_json.isnot(None))
        .order_by(Encounter.period_start.desc())
        .first()
    )
    if encounter is None:
        return None
    return CheckInResult(encounter=encounter, card=_load_card(encounter))


def search_encounters(db: Session, query: str) -> list[CheckInResult]:
    """ID search (PT-XXXX, exact) or a substring name search — mirrors V1's
    `_run_search`: try ID first, fall back to name."""
    query = query.strip()
    if query.upper().startswith(f"{PATIENT_ID_PREFIX}-"):
        result = get_encounter_by_patient_identifier(db, query)
        return [result] if result else []

    rows = (
        db.query(Encounter)
        .join(PatientRow, Encounter.patient_id == PatientRow.id)
        .filter(Encounter.card_json.isnot(None))
        .filter(func.lower(PatientRow.full_name).contains(query.lower()))
        .order_by(Encounter.period_start.desc())
        .all()
    )
    return [CheckInResult(encounter=e, card=_load_card(e)) for e in rows]


def get_prior_visits(db: Session, encounter_id: str) -> list[CheckInResult] | None:
    """
    Every other encounter belonging to the same Patient row, oldest first.
    Returns None if `encounter_id` itself doesn't exist, so the route can 404.

    A direct FK filter now that visits consolidate onto one Patient row —
    simpler and more correct than the earlier name-matching join, which was
    only ever a name comparison. Two Encounters sharing patient_id are
    guaranteed to be the same person by construction; two Encounters merely
    sharing a name string were only a heuristic.
    """
    current = db.get(Encounter, encounter_id)
    if current is None:
        return None

    rows = (
        db.query(Encounter)
        .filter(Encounter.patient_id == current.patient_id)
        .filter(Encounter.id != encounter_id, Encounter.card_json.isnot(None))
        .order_by(Encounter.period_start.asc())
        .all()
    )
    return [CheckInResult(encounter=e, card=_load_card(e)) for e in rows]


# ─── Disposition + Admin ───────────────────────────────────────────────────────

def resolve_encounter(db: Session, encounter_id: str, actor: User, note: str | None) -> CheckInResult | None:
    """A clinician's disposition action — closes out an encounter. Returns
    None if the encounter doesn't exist, so the route can 404."""
    encounter = db.get(Encounter, encounter_id)
    if encounter is None or encounter.card_json is None:
        return None

    card = _load_card(encounter)
    card.patient.status = TriageStatus.RESOLVED
    encounter.card_json = card.model_dump_json()
    encounter.status = EncounterStatus.FINISHED
    encounter.period_end = datetime.now(timezone.utc)

    audit_row = AuditLog(
        actor_user_id=actor.id,
        action=AuditAction.ESCALATION_RESOLVED,
        resource_type="encounter",
        resource_id=encounter.id,
    )
    audit_row.metadata_dict = {"note": note} if note else {}
    db.add(audit_row)

    db.commit()
    db.refresh(encounter)
    return CheckInResult(encounter=encounter, card=card)


def reset_all(db: Session, actor: User) -> None:
    """
    Admin end-of-shift reset. Wipes the clinical queue — Patient, Encounter,
    Observation, RiskAssessment — but never Users or AuditLog: the record of
    who reset the queue, and when, has to survive the reset itself, or the
    audit log's append-only promise means nothing at the one moment it
    matters most.
    """
    db.query(RiskAssessment).delete()
    db.query(ObservationRow).delete()
    db.query(Encounter).delete()
    db.query(PatientRow).delete()

    audit_row = AuditLog(
        actor_user_id=actor.id,
        action=AuditAction.QUEUE_RESET,
        resource_type="queue",
        resource_id="all",
    )
    db.add(audit_row)
    db.commit()
