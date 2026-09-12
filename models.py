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

class FindingSeverity(str, Enum):
    """How much weight a single deterministic finding carries."""
    CRITICAL   = "critical"    # warrants a physician now, independent of ESI
    CONCERNING = "concerning"  # outside normal range, raises concern


class FindingCategory(str, Enum):
    """What kind of check produced a finding."""
    VITAL   = "vital"
    SYMPTOM = "symptom"
    AGE     = "age"


class SymptomStatus(str, Enum):
    """
    What the chief complaint actually says about a symptom.

    A substring scan cannot tell these apart — "denies chest pain" and "chest
    pain" look identical to it. Structured extraction reports the distinction
    so only PRESENT symptoms become findings.
    """
    PRESENT    = "present"     # the patient has this now
    DENIED     = "denied"      # the complaint explicitly rules it out
    HISTORICAL = "historical"  # a past episode, not this presentation


class EscalationLevel(str, Enum):
    """
    How urgently this patient needs attention beyond their queue position.

    Escalation is derived from the ESI score *and* the deterministic findings.
    It is deliberately separate from the ESI score itself: every patient gets
    scored, and escalation is an additional signal layered on top.
    """
    NONE      = "none"       # routine — queue position governs
    ELEVATED  = "elevated"   # watch closely, re-assess sooner than queue order
    IMMEDIATE = "immediate"  # physician now


class TriageStatus(str, Enum):
    """Current processing status of the patient in the triage pipeline."""
    PENDING   = "pending"    # checked in, not yet scored
    TRIAGED   = "triaged"    # scored, no escalation
    ESCALATED = "escalated"  # scored and flagged for attention
    RESOLVED  = "resolved"   # clinician has dispositioned the patient


# ─── Clinical Reference ───────────────────────────────────────────────────────

class ReferenceChunk(BaseModel):
    """
    One retrievable section of the clinical reference.

    Chunks are split on markdown headings rather than a fixed character window,
    so each one is a self-contained criteria section and carries the heading it
    came from — which is what makes a citation in the prompt meaningful.
    """

    text: str
    source: str = Field(..., description="Source document filename")
    section: str = Field(..., description="Heading path this chunk came from")
    embedding: list[float] = Field(
        default_factory=list,
        description="Precomputed at build time; empty in a lexical-only index",
    )

    @property
    def citation(self) -> str:
        return f"{self.source} § {self.section}"


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
        """
        Collapse whitespace but preserve the clinician's original casing.

        Lowercasing here destroyed clinical abbreviations ("SOB" became "sob",
        and the UI's .title() then rendered it "Sob"). Matching lowercases at
        the point of comparison instead.
        """
        return " ".join(v.split())

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


class ExtractedSymptom(BaseModel):
    """
    One symptom the model recognised in the chief complaint, with its status.

    `symptom` is constrained to the vocabulary in config.py rather than free
    text, so severity mapping stays deterministic and auditable.
    """

    symptom: str        = Field(..., description="Term from the supplied symptom vocabulary")
    status: SymptomStatus = Field(..., description="Whether the complaint asserts, denies, or historicises it")

    def __str__(self) -> str:
        return f"{self.symptom} ({self.status.value})"


class ClinicalFinding(BaseModel):
    """A single deterministic finding produced by the rule-based assessment."""

    severity: FindingSeverity
    category: FindingCategory
    detail: str = Field(..., description="Human-readable finding with measured value")

    def __str__(self) -> str:
        return self.detail


class EscalationAssessment(BaseModel):
    """
    Escalation result — attached to every patient regardless of outcome.

    Note this does *not* replace the ESI score. A patient has both: an ESI level
    from the triage reasoning, and an escalation level derived from that score
    plus the deterministic findings below.
    """

    level: EscalationLevel               = EscalationLevel.NONE
    findings: list[ClinicalFinding]      = Field(default_factory=list)
    reasons: list[str]                   = Field(default_factory=list, description="Why this level was assigned")
    physician_summary: Optional[PhysicianSummary] = None
    system_error: Optional[str]          = Field(
        default=None,
        description="Set when escalation is the result of a pipeline failure rather than clinical judgment",
    )

    @property
    def triggered(self) -> bool:
        """True if this patient needs any attention beyond their queue position."""
        return self.level is not EscalationLevel.NONE

    @property
    def is_immediate(self) -> bool:
        return self.level is EscalationLevel.IMMEDIATE

    @property
    def critical_findings(self) -> list[ClinicalFinding]:
        return [f for f in self.findings if f.severity is FindingSeverity.CRITICAL]

    @property
    def abnormal_vitals(self) -> list[str]:
        """Vital-sign findings only, formatted for the physician summary."""
        return [f.detail for f in self.findings if f.category is FindingCategory.VITAL]


# ─── Triage Result ────────────────────────────────────────────────────────────

class TriageResult(BaseModel):
    """Output of the LLM triage reasoning node for non-flagged patients."""

    esi_score: int                     = Field(..., ge=1, le=5, description="ESI level 1–5")
    esi_rationale: str                 = Field(..., description="Why this ESI level was assigned")
    clinical_reasoning: str            = Field(..., description="Full clinical reasoning narrative")
    recommended_interventions: list[str] = Field(default_factory=list)
    retrieval_context_used: bool       = Field(default=True, description="Whether grounding context was actually retrieved")
    is_fallback: bool                  = Field(
        default=False,
        description="True when this score came from a failure path rather than clinical reasoning",
    )
    extracted_symptoms: list[ExtractedSymptom] = Field(
        default_factory=list,
        description="Symptoms recognised in the complaint, with present/denied/historical status. "
                    "Retained for audit and eval: it records what the model believed it read.",
    )

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
    escalation: EscalationAssessment       = Field(default_factory=EscalationAssessment)
    triage_result: Optional[TriageResult]  = None

    @property
    def display_esi(self) -> str:
        """
        ESI score as a display string.

        Every triaged patient has a score, including escalated ones — escalation
        is shown alongside the score, never instead of it.
        """
        if self.triage_result:
            return f"ESI {self.triage_result.esi_score}"
        if self.escalation.system_error:
            return "System Error"
        return "Pending"

    @property
    def esi_score(self) -> Optional[int]:
        return self.triage_result.esi_score if self.triage_result else None

    @property
    def is_escalated(self) -> bool:
        return self.escalation.triggered

    @property
    def needs_immediate_attention(self) -> bool:
        return self.escalation.is_immediate

    @property
    def has_system_error(self) -> bool:
        return self.escalation.system_error is not None


# ─── LangGraph State ──────────────────────────────────────────────────────────

class TriageState(TypedDict, total=False):
    """
    Shared state passed between all LangGraph nodes.

    Flow is linear — every patient traverses every node:
        assess → triage → escalate → build_card
    """

    patient: Patient
    vital_findings: list[ClinicalFinding]     # measured, from assess_node
    symptom_findings: list[ClinicalFinding]   # extracted, from triage_node
    findings: list[ClinicalFinding]           # combined + age risk, from escalate_node
    triage_result: Optional[TriageResult]     # ESI score, from triage_node
    escalation: Optional[EscalationAssessment]  # derived, from escalate_node
    patient_card: Optional[PatientCard]
    retrieval_context: str
    system_error: Optional[str]               # set when a node fails
