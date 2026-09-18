"""
backend/models/patient.py — FHIR Patient.

Deliberately narrow. FHIR's real Patient resource has far more than this
(HumanName with family/given parts, multiple identifiers and telecoms,
address, contact...). This captures what the app actually collects, using
FHIR's field names and semantics rather than inventing our own — that's the
"FHIR-shaped, not FHIR-certified" line drawn in V2_PLAN.md.

Two honest simplifications, stated here rather than left silent:
  - `full_name` is one field, not FHIR's structured HumanName. The intake
    form collects one string; splitting it into family/given without real
    data would be fabricating structure we don't have.
  - `birth_date` is derived from age-at-checkin (today's date minus age
    years), not a real date of birth — the app has never collected one.
    `birth_date_is_estimated` says so explicitly rather than presenting an
    estimate as fact.

Weight is intentionally NOT a field here — in real FHIR it's an Observation
(LOINC 29463-7, Body Weight), not a Patient attribute, and that's how it's
modeled in backend/models/observation.py.
"""

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # FHIR Patient.identifier — our human-readable chart number (PT-0001),
    # distinct from the internal UUID primary key.
    patient_identifier: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)

    # FHIR Patient.name — collapsed to one field; see module docstring.
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)

    # FHIR Patient.birthDate — estimated; see module docstring.
    birth_date: Mapped[date] = mapped_column(Date, nullable=False)
    birth_date_is_estimated: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # FHIR Patient.gender (administrative gender: male | female | other |
    # unknown). Not collected by the intake form — left null rather than
    # guessed. Matches FHIR's own 0..1 cardinality on this field.
    gender: Mapped[str | None] = mapped_column(String(10), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    @property
    def age_years(self) -> int:
        """Recompute age from the estimated birth date, for display."""
        today = date.today()
        years = today.year - self.birth_date.year
        if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
            years -= 1
        return years
