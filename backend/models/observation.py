"""
backend/models/observation.py — FHIR Observation.

One row per measurement, not one JSON blob per patient — that's how a real
EHR stores vitals, and it's what makes the audit trail and any future
trending query meaningful (e.g. "show me this patient's HR across visits").

LOINC codes below are the standard, widely-used codes for each measurement —
verify against an authoritative LOINC source before treating this as
clinically authoritative; this project does not claim full terminology
validation.
"""

import uuid
from datetime import datetime, timezone
from typing import NamedTuple

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class LoincCode(NamedTuple):
    code: str
    display: str
    unit: str


# FHIR Observation.code — maps each vital sign to its LOINC code, display
# name, and unit. This is the single source of truth for that mapping;
# services/ should reference this rather than hardcoding codes elsewhere.
VITAL_LOINC_CODES: dict[str, LoincCode] = {
    "heart_rate":        LoincCode("8867-4",  "Heart rate",                "bpm"),
    "respiratory_rate":  LoincCode("9279-1",  "Respiratory rate",          "breaths/min"),
    "spo2":              LoincCode("2708-6",  "Oxygen saturation",         "%"),
    "temperature_c":     LoincCode("8310-5",  "Body temperature",         "Cel"),
    "systolic_bp":       LoincCode("8480-6",  "Systolic blood pressure",   "mmHg"),
    "diastolic_bp":      LoincCode("8462-4",  "Diastolic blood pressure",  "mmHg"),
    "weight_kg":         LoincCode("29463-7", "Body weight",               "kg"),
}


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("patients.id"), nullable=False, index=True)
    encounter_id: Mapped[str] = mapped_column(String(36), ForeignKey("encounters.id"), nullable=False, index=True)

    code: Mapped[str] = mapped_column(String(20), nullable=False)          # LOINC code
    code_display: Mapped[str] = mapped_column(String(100), nullable=False)  # human label
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)

    effective_datetime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
