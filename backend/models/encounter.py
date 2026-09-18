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
