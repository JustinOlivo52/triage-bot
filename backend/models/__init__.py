"""
backend/models/__init__.py — imports every model so Base.metadata is
complete wherever Base is imported. Alembic's env.py relies on this: it
imports `backend.models` and nothing else to discover the full schema.
"""

from backend.models.audit_log import AuditAction, AuditLog
from backend.models.encounter import Encounter, EncounterStatus
from backend.models.observation import VITAL_LOINC_CODES, Observation
from backend.models.patient import Patient
from backend.models.risk_assessment import QUALITATIVE_RISK_BY_ESCALATION, RiskAssessment
from backend.models.user import User, UserRole

__all__ = [
    "AuditAction",
    "AuditLog",
    "Encounter",
    "EncounterStatus",
    "Observation",
    "VITAL_LOINC_CODES",
    "Patient",
    "RiskAssessment",
    "QUALITATIVE_RISK_BY_ESCALATION",
    "User",
    "UserRole",
]
