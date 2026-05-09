"""
models.py — Pydantic v2 data models for triage-bot.
Defines all patient, triage, red flag, and LangGraph state schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, field_validator


# ─── Enums ────────────────────────────────────────────────────────────────────

class RedFlagLayer(str, Enum):
    """Which evaluation layer triggered the red flag."""
    LAYER_1_ESI      = "ESI Score Critical (Level 1 or 2)"
    LAYER_2_CLINICAL = "Clinical Risk Factors (Age / Vitals / Symptoms)"
    LAYER_3_LLM      = "LLM Clinical Reasoning"


class TriageStatus(str, Enum):
    """Current processing status of the patient in the triage pipeline."""
    PENDING    = "pending"
    RED_FLAGGED = "red_flagged"
    TRIAGED    = "triaged"


# ─── Vitals ───────────────────────────────────────────────────────────────────

class VitalSigns(BaseModel):
    """Patient vital signs captured at check-in."""

    heart_rate: int        = Field(..., ge=0,   le=300,  description="Heart rate in bpm")
    systolic_bp: int       = Field(..., ge=0,   le=300,  description="Systolic blood pressure in mmHg")
    diastolic_bp: int      = Field(..., ge=0,   le=200,  description="Diastolic blood pressure in mmHg")
    respiratory_rate: int  = Field(..., ge=0,   le=60,   description="Respiratory rate in breaths/min")
    spo2: float            = Field(..., ge=0.0, le=100.0,description="Oxygen saturation as a percentage")
    temperature_c: float   = Field(..., ge=25.0,le=45.0, description="Temperature in Celsius")

    @field_validator("spo2")
    @classmethod
    def round_spo2(cls, v: float) -> float:
        """Keep SpO2 to one decimal place."""
        return round(v, 1)

    @field_validator("temperature_c")
    @classmethod
    def round_temp(cls, v: float) -> float:
        """Keep temperature to one decimal place."""
        return round(v, 1)

    @property
    def temperature_f(self) -> float:
        """Fahrenheit conversion for display."""
        return round((self.temperature_c * 9 / 5) + 32, 1)

    @property
    def bp_display(self) -> str:
        """Formatted blood pressure string."""
        return f"{self.systolic_bp}/{self.diastolic_bp} mmHg"


# ─── Patient ──────────────────────────────────────────────────────────────────

class Patient(BaseModel):
    """Core patient record created at check-in."""

    patient_id: str            = Field(..., description="Unique patient ID e.g. PT-0001")
    name: str                  = Field(..., min_length=1, description="Full name")
    age: int                   = Field(..., ge=0, le=130,  description="Age in years")
    weight_kg: float           = Field(..., gt=0, le=500,  description="Weight in kilograms")
    chief_complaint: str       = Field(..., min_length=1,  description="Patient's primary complaint in their own words")
    vitals: VitalSigns
    check_in_time: datetime    = Field(default_factory=datetime.now)
    is_returning: bool         = Field(default=False, description="True if patient has a prior visit in memory")
    status: TriageStatus       = Field(default=TriageStatus.PENDING)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Titlecase the patient name for consistent display."""
        return v.strip().title()

    @field_validator("chief_complaint")
    @classmethod
    def normalize_complaint(cls, v: str) -> str:
        """Strip and lowercase for consistent matching."""
        return v.strip().lower()

    @property
    def weight_lbs(self) -> float:
        """Pounds conversion for display."""
        return round(self.weight_kg * 2.205, 1)

    @property
    def age_group(self) -> str:
        """Broad age classification used in clinical risk logic."""
        if self.age <= 5:
            return "pediatric"
        if self.age >= 65:
            return "geriatric"
        return "adult"


# ─── Red Flag ─────────────────────────────────────────────────────────────────

class PhysicianSummary(BaseModel):
    """Structured alert summary delivered to the physician when a red flag fires."""

    urgency_statement: str         = Field(..., description="One-line urgency callout for the physician")
    trigger_reasons: list[str]     = Field(..., description="Bullet list of exactly why the flag fired")
    abnormal_vitals: list[str]     = Field(default_factory=list, description="List of out-of-range vitals with values")
    clinical_concerns: str         = Field(..., description="LLM-generated paragraph on the primary clinical concerns")
    recommended_actions: list[str] = Field(..., description="Immediate actions the physician should consider")


class RedFlagAlert(BaseModel):
    """Red flag evaluation result — attached to every patient regardless of outcome."""

    triggered: bool
    layer: Optional[RedFlagLayer]           = None
    reasons: list[str]                      = Field(default_factory=list)
    physician_summary: Optional[PhysicianSummary] = None


# ─── Triage Result ────────────────────────────────────────────────────────────

class TriageResult(BaseModel):
    """Output of the LLM triage reasoning node for non-flagged patients."""

    esi_score: int                     = Field(..., ge=1, le=5, description="ESI level 1–5")
    esi_rationale: str                 = Field(..., description="Why this ESI level was assigned")
    clinical_reasoning: str            = Field(..., description="Full clinical reasoning narrative")
    recommended_interventions: list[str] = Field(default_factory=list)
    retrieval_context_used: bool       = Field(default=True, description="Whether RAG context was retrieved")

    @field_validator("esi_score")
    @classmethod
    def validate_esi(cls, v: int) -> int:
        if v not in range(1, 6):
            raise ValueError("ESI score must be between 1 and 5")
        return v


# ─── Patient Card ─────────────────────────────────────────────────────────────

class PatientCard(BaseModel):
    """
    Final structured output presented to the healthcare team.
    Contains the full patient picture — vitals, complaint, ESI, red flag status.
    """

    patient: Patient
    red_flag_alert: RedFlagAlert           = Field(default_factory=lambda: RedFlagAlert(triggered=False))
    triage_result: Optional[TriageResult]  = None

    @property
    def display_esi(self) -> str:
        """ESI score as a display string, or alert if red flagged."""
        if self.red_flag_alert.triggered:
            return "RED FLAG — Physician Notified"
        if self.triage_result:
            return f"ESI {self.triage_result.esi_score}"
        return "Pending"

    @property
    def is_red_flagged(self) -> bool:
        return self.red_flag_alert.triggered


# ─── LangGraph State ──────────────────────────────────────────────────────────

class TriageState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.
    Each node reads from and writes back to this dict.
    """

    patient: Patient
    red_flag_alert: Optional[RedFlagAlert]
    triage_result: Optional[TriageResult]
    patient_card: Optional[PatientCard]
    retrieval_context: str
    esi_estimate: int
    error: Optional[str]
