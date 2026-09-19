"""
tests/backend/test_encounters.py — Check-in -> triage -> FHIR persistence,
the vertical slice from V2_PLAN.md Phase 2.

`run_triage` (the real LangGraph/Claude pipeline) is monkeypatched, the same
zero-API-key philosophy the rest of this project holds to. This tests the
service boundary — request in, FHIR rows out — not the LLM call itself,
which belongs to agents/ and needs a real key to exercise for real.
"""

import backend.services.triage_service as triage_service
from backend.models.audit_log import AuditAction, AuditLog
from backend.models.encounter import Encounter
from backend.models.observation import Observation
from backend.models.patient import Patient as PatientRow
from backend.models.risk_assessment import RiskAssessment
from backend.models.user import UserRole
from models import EscalationAssessment, EscalationLevel, Patient as DomainPatient, PatientCard, TriageResult
from tests.backend.conftest import auth_headers, make_user

VALID_VITALS = {
    "heart_rate": 88, "systolic_bp": 120, "diastolic_bp": 78,
    "respiratory_rate": 16, "spo2": 98.0, "temperature_c": 37.0,
}


def _check_in_payload(name="Jane Doe", complaint="denies chest pain, medication refill"):
    return {
        "full_name": name,
        "age": 40,
        "weight_kg": 70.0,
        "chief_complaint": complaint,
        "vitals": VALID_VITALS,
    }


def _stub_card(domain_patient: DomainPatient, esi: int, level: EscalationLevel) -> PatientCard:
    return PatientCard(
        patient=domain_patient,
        triage_result=TriageResult(
            esi_score=esi,
            esi_rationale="stub rationale",
            clinical_reasoning="stub clinical reasoning",
            recommended_interventions=["monitor"],
        ),
        escalation=EscalationAssessment(level=level, findings=[], reasons=[]),
    )


def _stub_run_triage(monkeypatch, esi: int = 3, level: EscalationLevel = EscalationLevel.NONE) -> list[DomainPatient]:
    """Replace the real pipeline with a deterministic stub. Returns the list
    of domain patients it was called with, so a test can inspect them."""
    calls: list[DomainPatient] = []

    def _fake(domain_patient: DomainPatient) -> PatientCard:
        calls.append(domain_patient)
        return _stub_card(domain_patient, esi=esi, level=level)

    monkeypatch.setattr(triage_service, "run_triage", _fake)
    return calls


def _stub_run_triage_system_error(monkeypatch) -> None:
    def _fake(domain_patient: DomainPatient) -> PatientCard:
        return PatientCard(
            patient=domain_patient,
            triage_result=None,
            escalation=EscalationAssessment(
                level=EscalationLevel.IMMEDIATE,
                findings=[],
                reasons=["System error — manual triage required"],
                system_error="RuntimeError: stubbed failure",
            ),
        )

    monkeypatch.setattr(triage_service, "run_triage", _fake)


class TestCheckIn:

    def test_requires_authentication(self, client, db_session):
        r = client.post("/encounters/check-in", json=_check_in_payload())
        assert r.status_code == 401

    def test_creates_fhir_rows_and_returns_the_card(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse1", UserRole.NURSE)
        headers = auth_headers(client, "nurse1")
        _stub_run_triage(monkeypatch, esi=3, level=EscalationLevel.NONE)

        r = client.post("/encounters/check-in", headers=headers, json=_check_in_payload())
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["card"]["triage_result"]["esi_score"] == 3
        assert body["encounter_id"]

        assert db_session.query(PatientRow).count() == 1
        assert db_session.query(Encounter).count() == 1
        assert db_session.query(Observation).count() == 7  # 6 vitals + weight

        risk = db_session.query(RiskAssessment).one()
        assert risk.prediction_outcome == "ESI 3"
        assert risk.qualitative_risk == "low"
        assert risk.performer_model  # not empty — records which model made the call

        actions = {row.action for row in db_session.query(AuditLog).all()}
        assert AuditAction.PATIENT_CHECK_IN in actions
        assert AuditAction.TRIAGE_RUN in actions

    def test_immediate_escalation_maps_to_high_qualitative_risk(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse2", UserRole.NURSE)
        headers = auth_headers(client, "nurse2")
        _stub_run_triage(monkeypatch, esi=1, level=EscalationLevel.IMMEDIATE)

        r = client.post("/encounters/check-in", headers=headers, json=_check_in_payload())
        assert r.status_code == 201
        risk = db_session.query(RiskAssessment).one()
        assert risk.qualitative_risk == "high"

    def test_system_error_writes_no_risk_assessment(self, client, db_session, monkeypatch):
        """A failed triage call must not be persisted as a fabricated score."""
        make_user(db_session, "nurse3", UserRole.NURSE)
        headers = auth_headers(client, "nurse3")
        _stub_run_triage_system_error(monkeypatch)

        r = client.post("/encounters/check-in", headers=headers, json=_check_in_payload())
        assert r.status_code == 201
        assert r.json()["card"]["triage_result"] is None

        assert db_session.query(RiskAssessment).count() == 0
        actions = [row.action for row in db_session.query(AuditLog).all()]
        assert AuditAction.TRIAGE_SYSTEM_ERROR in actions
        assert AuditAction.TRIAGE_RUN not in actions

    def test_patient_identifiers_increment(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse4", UserRole.NURSE)
        headers = auth_headers(client, "nurse4")
        _stub_run_triage(monkeypatch)

        r1 = client.post("/encounters/check-in", headers=headers, json=_check_in_payload(name="Pat One"))
        r2 = client.post("/encounters/check-in", headers=headers, json=_check_in_payload(name="Pat Two"))

        assert r1.json()["card"]["patient"]["patient_id"] == "PT-0001"
        assert r2.json()["card"]["patient"]["patient_id"] == "PT-0002"

    def test_is_returning_detected_by_name(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse5", UserRole.NURSE)
        headers = auth_headers(client, "nurse5")
        calls = _stub_run_triage(monkeypatch)

        client.post("/encounters/check-in", headers=headers, json=_check_in_payload(name="Sam Returning"))
        client.post("/encounters/check-in", headers=headers, json=_check_in_payload(name="Sam Returning"))

        assert calls[0].is_returning is False
        assert calls[1].is_returning is True

    def test_returning_patient_reuses_the_same_patient_record(self, client, db_session, monkeypatch):
        """One Patient row, one patient_identifier, across visits — not a
        fresh chart number every time the same person checks in."""
        make_user(db_session, "nurse6", UserRole.NURSE)
        headers = auth_headers(client, "nurse6")
        _stub_run_triage(monkeypatch)

        r1 = client.post("/encounters/check-in", headers=headers, json=_check_in_payload(name="Consolidated Patient"))
        r2 = client.post("/encounters/check-in", headers=headers, json=_check_in_payload(name="Consolidated Patient"))

        id1 = r1.json()["card"]["patient"]["patient_id"]
        id2 = r2.json()["card"]["patient"]["patient_id"]
        assert id1 == id2
        assert r1.json()["encounter_id"] != r2.json()["encounter_id"]

        assert db_session.query(PatientRow).count() == 1
        assert db_session.query(Encounter).count() == 2

    def test_returning_patient_keeps_original_birth_date_but_new_age_at_encounter(
        self, client, db_session, monkeypatch
    ):
        """A person's birth date doesn't change between visits, even if the
        age they report does (a birthday, or a typo corrected next time)."""
        make_user(db_session, "nurse7", UserRole.NURSE)
        headers = auth_headers(client, "nurse7")
        _stub_run_triage(monkeypatch)

        client.post("/encounters/check-in", headers=headers, json=_check_in_payload(name="Birthday Patient"))
        patient_row = db_session.query(PatientRow).filter(PatientRow.full_name == "Birthday Patient").one()
        original_birth_date = patient_row.birth_date

        older_payload = _check_in_payload(name="Birthday Patient")
        older_payload["age"] = 41
        client.post("/encounters/check-in", headers=headers, json=older_payload)

        db_session.refresh(patient_row)
        assert patient_row.birth_date == original_birth_date
        assert db_session.query(PatientRow).count() == 1

        encounters = (
            db_session.query(Encounter)
            .filter(Encounter.patient_id == patient_row.id)
            .order_by(Encounter.period_start)
            .all()
        )
        assert [e.age_at_encounter for e in encounters] == [40, 41]

    def test_other_staff_roles_can_also_check_in(self, client, db_session, monkeypatch):
        """Check-in isn't a privileged action — every staff role can do it."""
        make_user(db_session, "physician1", UserRole.PHYSICIAN)
        headers = auth_headers(client, "physician1")
        _stub_run_triage(monkeypatch)

        r = client.post("/encounters/check-in", headers=headers, json=_check_in_payload())
        assert r.status_code == 201
