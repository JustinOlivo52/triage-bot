"""
backend/models/encounter.py — FHIR Encounter.

One row per ED visit. `class_code` uses the real HL7 v3 ActEncounterCode
value FHIR's Encounter.class references — "EMER" for emergency — rather than
inventing our own vocabulary for something FHIR already standardizes.
"""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class EncounterStatus(str, enum.Enum):
    """Subset of FHIR Encounter.status relevant to an ED visit's lifecycle."""
    IN_PROGRESS = "in-progress"
    FINISHED = "finished"


class Encounter(Base):
    __tablename__ = "encounters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("patients.id"), nullable=False, index=True)

    # FHIR Encounter.class — fixed at "EMER" (emergency); this app only
    # models the ED, so there is nothing else it could be yet.
    class_code: Mapped[str] = mapped_column(String(10), default="EMER", nullable=False)

    status: Mapped[EncounterStatus] = mapped_column(
        Enum(EncounterStatus), default=EncounterStatus.IN_PROGRESS, nullable=False
    )

    # FHIR Encounter.reasonCode — real FHIR uses a CodeableConcept (coded +
    # text); this is free text, because that's what the intake form actually
    # collects and coding it would mean inventing a terminology binding we
    # don't have.
    reason_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Not a FHIR field — carried because the pipeline's age-banding logic
    # (pediatric/geriatric risk amplification) needs age *at this visit*,
    # which the estimated Patient.birth_date alone doesn't cleanly give you
    # if a patient returns across a birthday.
    age_at_encounter: Mapped[int] = mapped_column(Integer, nullable=False)

    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Not a FHIR field. The FHIR rows above (this Encounter, its Observations,
    # its RiskAssessment) are the structured, queryable decomposition of a
    # visit — but the pipeline's native PatientCard carries things no FHIR
    # resource here has a slot for (recommended_interventions, extracted
    # symptoms with present/denied/historical status, the structured physician
    # summary, the escalation reasons list). Re-deriving all of that by
    # parsing RiskAssessment.basis/rationale text back into structure would be
    # fragile and pointless when the structured object already exists at
    # persistence time. This column caches that object verbatim, serving the
    # UI, while the FHIR rows remain the resource-shaped source of truth for
    # anything that queries by LOINC code, risk level, etc.
    card_json: Mapped[str | None] = mapped_column(Text, nullable=True)
