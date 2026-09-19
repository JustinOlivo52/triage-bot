"""
backend/models/risk_assessment.py — FHIR RiskAssessment.

This is the real FHIR resource for "a clinical judgment about the level of
risk of a patient for a domain" — which is exactly what an ESI triage score
is. Reaching for RiskAssessment instead of inventing a custom resource is
the whole point of going FHIR-shaped: an EHR-familiar reader recognizes the
mapping immediately.

`qualitative_risk` uses FHIR's own risk-probability values (a real,
standard value set: negligible | low | moderate | high | certain) — our
escalation levels map onto a subset of it rather than getting their own
vocabulary:

    EscalationLevel.NONE      -> "low"
    EscalationLevel.ELEVATED  -> "moderate"
    EscalationLevel.IMMEDIATE -> "high"
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base

# See module docstring — maps our EscalationLevel values onto FHIR's
# risk-probability value set. Kept here rather than importing models.py's
# EscalationLevel directly, so backend/ has no import-time dependency on the
# pipeline's enum shape.
QUALITATIVE_RISK_BY_ESCALATION: dict[str, str] = {
    "none": "low",
    "elevated": "moderate",
    "immediate": "high",
}


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("patients.id"), nullable=False, index=True)
    encounter_id: Mapped[str] = mapped_column(String(36), ForeignKey("encounters.id"), nullable=False, index=True)

    # FHIR RiskAssessment.prediction.outcome — the ESI level, as text ("ESI 2").
    # No score is None here: the pipeline never leaves this pipeline without
    # one, and a system-error card is handled as its own audit event, not a
    # RiskAssessment row with a fabricated outcome.
    prediction_outcome: Mapped[str] = mapped_column(String(20), nullable=False)

    # FHIR RiskAssessment.prediction.qualitativeRisk — see module docstring.
    qualitative_risk: Mapped[str] = mapped_column(String(20), nullable=False)

    # FHIR RiskAssessment.basis — what the judgment was based on. Stored as
    # the same finding-detail strings agents/assessment.py already produces,
    # joined into one text block rather than a separate table, since this
    # exists for audit/read purposes, not for querying individual findings.
    basis: Mapped[str] = mapped_column(Text, nullable=False)

    # FHIR RiskAssessment.rationale — the clinical reasoning narrative.
    rationale: Mapped[str] = mapped_column(Text, nullable=False)

    # Not a FHIR field, but real FHIR RiskAssessment does have a
    # `performer` reference — here that performer is the model, not a
    # clinician, and saying so plainly is a form of the same honesty this
    # whole model exercises: it distinguishes AI-generated risk assessment
    # from clinician judgment rather than blurring the two.
    performer_model: Mapped[str] = mapped_column(String(50), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
