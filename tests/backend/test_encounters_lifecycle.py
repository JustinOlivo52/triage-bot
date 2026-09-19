"""
tests/backend/test_encounters_lifecycle.py — Reading the queue, searching,
prior visits, disposition (resolve), and the admin reset: the rest of the
encounter lifecycle main.py needs once it stops importing memory/ and
agents/ directly (V2_PLAN.md Phase 4).
"""

import backend.services.triage_service as triage_service
from backend.models.audit_log import AuditAction, AuditLog
from backend.models.encounter import Encounter, EncounterStatus
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


def _payload(name="Jane Doe", complaint="ankle sprain"):
    return {
        "full_name": name, "age": 40, "weight_kg": 70.0,
        "chief_complaint": complaint, "vitals": VALID_VITALS,
    }


def _stub_run_triage(monkeypatch, esi=4, level=EscalationLevel.NONE):
    def _fake(domain_patient: DomainPatient) -> PatientCard:
        return PatientCard(
            patient=domain_patient,
            triage_result=TriageResult(
                esi_score=esi, esi_rationale="r", clinical_reasoning="c",
            ),
            escalation=EscalationAssessment(level=level),
        )

    monkeypatch.setattr(triage_service, "run_triage", _fake)


class TestListAndGet:

    def test_list_returns_checked_in_patients(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse1", UserRole.NURSE)
        headers = auth_headers(client, "nurse1")
        _stub_run_triage(monkeypatch)

        client.post("/encounters/check-in", headers=headers, json=_payload(name="A"))
        client.post("/encounters/check-in", headers=headers, json=_payload(name="B"))

        r = client.get("/encounters", headers=headers)
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_active_only_excludes_resolved(self, client, db_session, monkeypatch):
        make_user(db_session, "physician1", UserRole.PHYSICIAN)
        headers = auth_headers(client, "physician1")
        _stub_run_triage(monkeypatch)

        r = client.post("/encounters/check-in", headers=headers, json=_payload())
        encounter_id = r.json()["encounter_id"]
        client.post(f"/encounters/{encounter_id}/resolve", headers=headers, json={})

        assert len(client.get("/encounters", headers=headers).json()) == 1
        assert len(client.get("/encounters?active_only=true", headers=headers).json()) == 0

    def test_get_by_patient_identifier(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse2", UserRole.NURSE)
        headers = auth_headers(client, "nurse2")
        _stub_run_triage(monkeypatch)

        r = client.post("/encounters/check-in", headers=headers, json=_payload())
        patient_id = r.json()["card"]["patient"]["patient_id"]

        got = client.get(f"/encounters/by-patient/{patient_id}", headers=headers)
        assert got.status_code == 200
        assert got.json()["card"]["patient"]["patient_id"] == patient_id

    def test_get_by_unknown_patient_identifier_is_404(self, client, db_session):
        make_user(db_session, "nurse3", UserRole.NURSE)
        headers = auth_headers(client, "nurse3")
        r = client.get("/encounters/by-patient/PT-9999", headers=headers)
        assert r.status_code == 404


class TestSearch:

    def test_search_by_id(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse4", UserRole.NURSE)
        headers = auth_headers(client, "nurse4")
        _stub_run_triage(monkeypatch)

        r = client.post("/encounters/check-in", headers=headers, json=_payload())
        patient_id = r.json()["card"]["patient"]["patient_id"]

        found = client.get("/encounters/search", headers=headers, params={"q": patient_id})
        assert found.status_code == 200
        assert len(found.json()) == 1

    def test_search_by_name_substring(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse5", UserRole.NURSE)
        headers = auth_headers(client, "nurse5")
        _stub_run_triage(monkeypatch)

        client.post("/encounters/check-in", headers=headers, json=_payload(name="Samantha Reyes"))
        client.post("/encounters/check-in", headers=headers, json=_payload(name="Bob Fields"))

        found = client.get("/encounters/search", headers=headers, params={"q": "reyes"})
        assert len(found.json()) == 1
        assert found.json()[0]["card"]["patient"]["name"] == "Samantha Reyes"


class TestPriorVisits:

    def test_returns_earlier_encounters_for_the_same_name(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse6", UserRole.NURSE)
        headers = auth_headers(client, "nurse6")
        _stub_run_triage(monkeypatch)

        r1 = client.post("/encounters/check-in", headers=headers, json=_payload(name="Repeat Visitor"))
        r2 = client.post("/encounters/check-in", headers=headers, json=_payload(name="Repeat Visitor"))

        prior = client.get(f"/encounters/{r2.json()['encounter_id']}/prior-visits", headers=headers)
        assert prior.status_code == 200
        assert len(prior.json()) == 1
        assert prior.json()[0]["encounter_id"] == r1.json()["encounter_id"]

    def test_unknown_encounter_is_404(self, client, db_session):
        make_user(db_session, "nurse7", UserRole.NURSE)
        headers = auth_headers(client, "nurse7")
        r = client.get("/encounters/not-a-real-id/prior-visits", headers=headers)
        assert r.status_code == 404


class TestResolve:

    def test_physician_can_resolve(self, client, db_session, monkeypatch):
        make_user(db_session, "physician2", UserRole.PHYSICIAN)
        headers = auth_headers(client, "physician2")
        _stub_run_triage(monkeypatch)

        r = client.post("/encounters/check-in", headers=headers, json=_payload())
        encounter_id = r.json()["encounter_id"]

        resolved = client.post(
            f"/encounters/{encounter_id}/resolve", headers=headers, json={"note": "discharged home"}
        )
        assert resolved.status_code == 200
        assert resolved.json()["card"]["patient"]["status"] == "resolved"

        row = db_session.get(Encounter, encounter_id)
        assert row.status is EncounterStatus.FINISHED
        assert row.period_end is not None

        audit_rows = db_session.query(AuditLog).filter(AuditLog.action == AuditAction.ESCALATION_RESOLVED).all()
        assert len(audit_rows) == 1
        assert audit_rows[0].metadata_dict["note"] == "discharged home"

    def test_nurse_cannot_resolve(self, client, db_session, monkeypatch):
        make_user(db_session, "nurse8", UserRole.NURSE)
        headers = auth_headers(client, "nurse8")
        _stub_run_triage(monkeypatch)

        r = client.post("/encounters/check-in", headers=headers, json=_payload())
        encounter_id = r.json()["encounter_id"]

        resolved = client.post(f"/encounters/{encounter_id}/resolve", headers=headers, json={})
        assert resolved.status_code == 403

    def test_resolving_unknown_encounter_is_404(self, client, db_session):
        make_user(db_session, "physician3", UserRole.PHYSICIAN)
        headers = auth_headers(client, "physician3")
        r = client.post("/encounters/not-a-real-id/resolve", headers=headers, json={})
        assert r.status_code == 404


class TestReset:

    def test_admin_can_reset_and_audit_log_survives(self, client, db_session, monkeypatch):
        make_user(db_session, "admin4", UserRole.ADMIN)
        headers = auth_headers(client, "admin4")
        _stub_run_triage(monkeypatch)

        client.post("/encounters/check-in", headers=headers, json=_payload())
        audit_rows_before = db_session.query(AuditLog).count()

        r = client.post("/encounters/reset", headers=headers)
        assert r.status_code == 204

        assert db_session.query(PatientRow).count() == 0
        assert db_session.query(Encounter).count() == 0
        assert db_session.query(Observation).count() == 0
        assert db_session.query(RiskAssessment).count() == 0
        # Reset never deletes the audit trail, including the record of the
        # reset itself.
        assert db_session.query(AuditLog).count() == audit_rows_before + 1
        assert AuditAction.QUEUE_RESET in {r.action for r in db_session.query(AuditLog).all()}

    def test_nurse_cannot_reset(self, client, db_session):
        make_user(db_session, "nurse9", UserRole.NURSE)
        headers = auth_headers(client, "nurse9")
        r = client.post("/encounters/reset", headers=headers)
        assert r.status_code == 403

    def test_physician_cannot_reset(self, client, db_session):
        """Resolving a patient and wiping the whole department are different
        axes of privilege — physician gets the first, not the second."""
        make_user(db_session, "physician4", UserRole.PHYSICIAN)
        headers = auth_headers(client, "physician4")
        r = client.post("/encounters/reset", headers=headers)
        assert r.status_code == 403
