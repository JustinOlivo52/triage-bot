"""
tests/backend/test_models.py — SQLAlchemy models round-trip correctly and
the FHIR mapping tables (LOINC codes, risk-probability values) are what the
code and docs claim they are.
"""

from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError

from backend.models.audit_log import AuditAction, AuditLog
from backend.models.encounter import Encounter, EncounterStatus
from backend.models.observation import VITAL_LOINC_CODES, Observation
from backend.models.patient import Patient
from backend.models.risk_assessment import QUALITATIVE_RISK_BY_ESCALATION, RiskAssessment
from backend.models.user import User, UserRole
from backend.core.security import hash_password


class TestPatient:

    def test_round_trips(self, db_session):
        p = Patient(
            patient_identifier="PT-0001",
            full_name="Test Patient",
            birth_date=date(1990, 1, 1),
            birth_date_is_estimated=True,
        )
        db_session.add(p)
        db_session.commit()
        db_session.refresh(p)

        fetched = db_session.get(Patient, p.id)
        assert fetched.patient_identifier == "PT-0001"
        assert fetched.gender is None  # not collected — must be None, not guessed

    def test_age_years_computed_from_birth_date(self, db_session):
        today = date.today()
        birth_date = date(today.year - 40, today.month, today.day)
        p = Patient(patient_identifier="PT-0002", full_name="X", birth_date=birth_date)
        assert p.age_years == 40

    def test_patient_identifier_must_be_unique(self, db_session):
        db_session.add(Patient(patient_identifier="PT-0003", full_name="A", birth_date=date(2000, 1, 1)))
        db_session.commit()

        db_session.add(Patient(patient_identifier="PT-0003", full_name="B", birth_date=date(2000, 1, 1)))
        import pytest
        with pytest.raises(IntegrityError):
            db_session.commit()


class TestEncounter:

    def test_defaults_to_emergency_class_and_in_progress(self, db_session):
        p = Patient(patient_identifier="PT-0004", full_name="X", birth_date=date(2000, 1, 1))
        db_session.add(p)
        db_session.commit()

        e = Encounter(patient_id=p.id, reason_text="chest pain", age_at_encounter=45)
        db_session.add(e)
        db_session.commit()
        db_session.refresh(e)

        assert e.class_code == "EMER"
        assert e.status is EncounterStatus.IN_PROGRESS
        assert e.period_end is None


class TestObservation:

    def test_loinc_table_covers_every_vital_sign(self):
        """
        Regression guard: if VitalSigns in models.py gains a field, this
        table must be updated too, or that vital silently has no LOINC
        mapping when persisted.
        """
        expected = {
            "heart_rate", "respiratory_rate", "spo2",
            "temperature_c", "systolic_bp", "diastolic_bp", "weight_kg",
        }
        assert set(VITAL_LOINC_CODES.keys()) == expected

    def test_loinc_codes_are_well_formed(self):
        """LOINC codes are digits-dash-checkdigit, e.g. '8867-4'."""
        import re
        pattern = re.compile(r"^\d{1,6}-\d$")
        for key, entry in VITAL_LOINC_CODES.items():
            assert pattern.match(entry.code), f"{key}: {entry.code!r} doesn't look like a LOINC code"

    def test_round_trips(self, db_session):
        p = Patient(patient_identifier="PT-0005", full_name="X", birth_date=date(2000, 1, 1))
        db_session.add(p)
        db_session.commit()
        e = Encounter(patient_id=p.id, reason_text="x", age_at_encounter=30)
        db_session.add(e)
        db_session.commit()

        hr = VITAL_LOINC_CODES["heart_rate"]
        obs = Observation(
            patient_id=p.id, encounter_id=e.id,
            code=hr.code, code_display=hr.display, value=88, unit=hr.unit,
        )
        db_session.add(obs)
        db_session.commit()
        db_session.refresh(obs)

        assert obs.code == "8867-4"
        assert obs.value == 88


class TestRiskAssessment:

    def test_qualitative_risk_mapping_is_a_real_fhir_value_set(self):
        """
        FHIR's risk-probability value set is exactly:
        negligible | low | moderate | high | certain.
        Our three escalation levels must map onto values from that set, not
        an invented vocabulary.
        """
        fhir_risk_probability_values = {"negligible", "low", "moderate", "high", "certain"}
        assert set(QUALITATIVE_RISK_BY_ESCALATION.values()) <= fhir_risk_probability_values

    def test_maps_all_three_escalation_levels(self):
        assert set(QUALITATIVE_RISK_BY_ESCALATION.keys()) == {"none", "elevated", "immediate"}

    def test_round_trips(self, db_session):
        p = Patient(patient_identifier="PT-0006", full_name="X", birth_date=date(2000, 1, 1))
        db_session.add(p)
        db_session.commit()
        e = Encounter(patient_id=p.id, reason_text="x", age_at_encounter=30)
        db_session.add(e)
        db_session.commit()

        ra = RiskAssessment(
            patient_id=p.id, encounter_id=e.id,
            prediction_outcome="ESI 2", qualitative_risk="high",
            basis="Marked tachycardia", rationale="Sepsis physiology suspected.",
            performer_model="claude-opus-5",
        )
        db_session.add(ra)
        db_session.commit()
        db_session.refresh(ra)

        assert ra.prediction_outcome == "ESI 2"
        assert ra.performer_model == "claude-opus-5"


class TestAuditLog:

    def test_metadata_dict_round_trips_through_json(self, db_session):
        u = User(username="a", hashed_password=hash_password("x"), full_name="A", role=UserRole.NURSE)
        db_session.add(u)
        db_session.commit()

        log = AuditLog(
            actor_user_id=u.id, action=AuditAction.LOGIN,
            resource_type="user", resource_id=u.id,
        )
        log.metadata_dict = {"ip": "127.0.0.1", "count": 3}
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        assert log.metadata_dict == {"ip": "127.0.0.1", "count": 3}

    def test_default_metadata_is_empty_dict_not_none(self, db_session):
        u = User(username="b", hashed_password=hash_password("x"), full_name="B", role=UserRole.NURSE)
        db_session.add(u)
        db_session.commit()

        log = AuditLog(actor_user_id=u.id, action=AuditAction.LOGIN, resource_type="user", resource_id=u.id)
        db_session.add(log)
        db_session.commit()
        db_session.refresh(log)

        assert log.metadata_dict == {}

    def test_rows_cannot_be_updated_at_the_database_level(self, db_session):
        """Not just 'no update route exists' — the database itself refuses,
        via the BEFORE UPDATE trigger registered in audit_log.py."""
        u = User(username="c", hashed_password=hash_password("x"), full_name="C", role=UserRole.NURSE)
        db_session.add(u)
        db_session.commit()

        log = AuditLog(actor_user_id=u.id, action=AuditAction.LOGIN, resource_type="user", resource_id=u.id)
        db_session.add(log)
        db_session.commit()

        log.resource_type = "tampered"
        with pytest.raises(IntegrityError, match="append-only"):
            db_session.commit()
        db_session.rollback()

    def test_rows_cannot_be_deleted_at_the_database_level(self, db_session):
        u = User(username="d", hashed_password=hash_password("x"), full_name="D", role=UserRole.NURSE)
        db_session.add(u)
        db_session.commit()

        log = AuditLog(actor_user_id=u.id, action=AuditAction.LOGIN, resource_type="user", resource_id=u.id)
        db_session.add(log)
        db_session.commit()

        db_session.delete(log)
        with pytest.raises(IntegrityError, match="append-only"):
            db_session.commit()
        db_session.rollback()

    def test_bulk_delete_is_also_blocked(self, db_session):
        """Triggers are row-level, so a bulk DELETE can't route around the
        single-row check above."""
        u = User(username="e", hashed_password=hash_password("x"), full_name="E", role=UserRole.NURSE)
        db_session.add(u)
        db_session.commit()

        db_session.add(AuditLog(actor_user_id=u.id, action=AuditAction.LOGIN, resource_type="user", resource_id=u.id))
        db_session.commit()

        with pytest.raises(IntegrityError, match="append-only"):
            db_session.query(AuditLog).delete()
            db_session.commit()
        db_session.rollback()
