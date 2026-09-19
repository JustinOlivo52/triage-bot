"""
tests/backend/test_demo_seed.py — Fixed demo accounts and the seed cohort,
loaded straight from the real committed data/seed_cohort.json. No LLM key
or network needed: that file is real model output already on disk, and
seeding it back in is pure persistence, same as any other check-in.
"""

from backend.models.audit_log import AuditLog
from backend.models.patient import Patient as PatientRow
from backend.models.risk_assessment import RiskAssessment
from backend.models.user import User, UserRole
from backend.services.demo_seed import run_demo_seed, seed_demo_accounts, seed_demo_cohort
from backend.services.triage_service import list_encounters


class TestSeedDemoAccounts:

    def test_creates_all_three_roles(self, db_session):
        seed_demo_accounts(db_session)
        usernames = {u.username: u.role for u in db_session.query(User).all()}
        assert usernames["demo_nurse"] is UserRole.NURSE
        assert usernames["demo_physician"] is UserRole.PHYSICIAN
        assert usernames["demo_admin"] is UserRole.ADMIN

    def test_is_idempotent(self, db_session):
        seed_demo_accounts(db_session)
        seed_demo_accounts(db_session)
        assert db_session.query(User).count() == 3

    def test_returns_the_admin_account(self, db_session):
        admin = seed_demo_accounts(db_session)
        assert admin.role is UserRole.ADMIN
        assert admin.username == "demo_admin"


class TestSeedDemoCohort:

    def test_loads_all_eight_seeded_patients(self, db_session):
        admin = seed_demo_accounts(db_session)
        seed_demo_cohort(db_session, admin)

        assert db_session.query(PatientRow).count() == 8
        results = list_encounters(db_session)
        assert len(results) == 8
        assert db_session.query(RiskAssessment).count() == 8  # every seeded patient has a score

    def test_writes_a_check_in_audit_row_per_seeded_patient(self, db_session):
        admin = seed_demo_accounts(db_session)
        seed_demo_cohort(db_session, admin)
        assert db_session.query(AuditLog).count() == 8

    def test_does_not_reseed_over_a_real_check_in(self, db_session, monkeypatch):
        import backend.services.triage_service as triage_service
        from models import EscalationAssessment, EscalationLevel, PatientCard, TriageResult

        def _fake(domain_patient):
            return PatientCard(
                patient=domain_patient,
                triage_result=TriageResult(
                    esi_score=4, esi_rationale="r", clinical_reasoning="c",
                ),
                escalation=EscalationAssessment(level=EscalationLevel.NONE),
            )

        monkeypatch.setattr(triage_service, "run_triage", _fake)

        from backend.schemas.encounter import CheckInRequest
        from models import VitalSigns
        admin = seed_demo_accounts(db_session)
        triage_service.check_in_and_triage(
            db_session,
            CheckInRequest(
                full_name="Real Patient", age=30, weight_kg=70.0,
                chief_complaint="test",
                vitals=VitalSigns(
                    heart_rate=80, systolic_bp=120, diastolic_bp=80,
                    respiratory_rate=16, spo2=98.0, temperature_c=37.0,
                ),
            ),
            admin,
        )

        seed_demo_cohort(db_session, admin)  # must be a no-op now
        assert db_session.query(PatientRow).count() == 1

    def test_run_demo_seed_does_both_steps(self, db_session):
        run_demo_seed(db_session)
        assert db_session.query(User).count() == 3
        assert db_session.query(PatientRow).count() == 8
