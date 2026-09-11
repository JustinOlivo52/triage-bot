"""
agents/escalation.py — Escalation assembly and physician alert generation.

The deterministic rules live in `agents/assessment.py` and have no LLM
dependency. This module is the thin layer that needs one:

    build_escalation()           → EscalationAssessment  (orchestration)
    generate_physician_summary() → PhysicianSummary      (LLM, IMMEDIATE only)

Only IMMEDIATE escalations trigger an LLM call, so routine and elevated
patients cost nothing beyond their triage scoring.
"""

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from config import ANTHROPIC_API_KEY, SUMMARY_MODEL
from models import (
    ClinicalFinding,
    EscalationAssessment,
    EscalationLevel,
    FindingCategory,
    Patient,
    PhysicianSummary,
)
from agents.assessment import derive_escalation

logger = logging.getLogger(__name__)


# ─── LLM Output Schema ────────────────────────────────────────────────────────

class _SummaryContent(BaseModel):
    """Structured output schema for the physician summary LLM call."""
    urgency_statement: str         = Field(description="One-line urgency callout for the physician")
    trigger_reasons: list[str]     = Field(description="Exactly why this patient was escalated")
    abnormal_vitals: list[str]     = Field(description="Out-of-range vitals with measured values")
    clinical_concerns: str         = Field(description="Paragraph summarizing the primary clinical concerns")
    recommended_actions: list[str] = Field(description="Immediate actions the physician should consider")


def _get_llm() -> ChatAnthropic:
    """Return the Claude instance used for physician alert narrative."""
    return ChatAnthropic(
        model=SUMMARY_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )


# ─── Escalation Assembly ──────────────────────────────────────────────────────

def build_escalation(
    patient: Patient,
    esi_score: Optional[int],
    findings: list[ClinicalFinding],
    system_error: Optional[str] = None,
) -> EscalationAssessment:
    """
    Assemble the full escalation assessment, generating a physician summary
    only when the level is IMMEDIATE.

    A system error always escalates to IMMEDIATE so nobody is silently dropped,
    but it is recorded separately so the UI can say "the pipeline failed" rather
    than implying a clinical judgment was made.
    """
    if system_error is not None:
        logger.error("Escalating %s due to system error: %s", patient.patient_id, system_error)
        return EscalationAssessment(
            level=EscalationLevel.IMMEDIATE,
            findings=findings,
            reasons=[f"Automated triage could not complete: {system_error}"],
            system_error=system_error,
            physician_summary=PhysicianSummary(
                urgency_statement=(
                    f"Manual triage required for {patient.patient_id} — automated assessment failed."
                ),
                trigger_reasons=[f"Pipeline error: {system_error}"],
                abnormal_vitals=[f.detail for f in findings if f.category is FindingCategory.VITAL],
                clinical_concerns=(
                    "The automated pipeline did not produce a triage decision for this patient. "
                    "No clinical judgment has been applied. Triage manually."
                ),
                recommended_actions=["Triage this patient manually at the bedside."],
            ),
        )

    level, reasons = derive_escalation(esi_score, findings)

    summary: Optional[PhysicianSummary] = None
    if level is EscalationLevel.IMMEDIATE:
        summary = generate_physician_summary(patient, esi_score, reasons, findings)

    logger.info(
        "Escalation for %s — level=%s, ESI=%s",
        patient.patient_id, level.value, esi_score,
    )
    return EscalationAssessment(
        level=level,
        findings=findings,
        reasons=reasons,
        physician_summary=summary,
    )


# ─── Physician Summary ────────────────────────────────────────────────────────

def generate_physician_summary(
    patient: Patient,
    esi_score: Optional[int],
    reasons: list[str],
    findings: list[ClinicalFinding],
) -> PhysicianSummary:
    """
    Generate a structured physician alert. Called only for IMMEDIATE escalation.

    The patient's name is deliberately not sent — it does not inform the alert.
    """
    abnormal_vitals = [f.detail for f in findings if f.category is FindingCategory.VITAL]

    system_prompt = """You are generating a concise, actionable alert summary for an emergency physician.
The patient has been escalated by the triage system and requires immediate attention.

Write with clinical precision. Be direct. Avoid filler language.
Recommended actions should be specific and immediately actionable."""

    alert_input = f"""
ESCALATION — IMMEDIATE

PATIENT:
- ID: {patient.patient_id}
- Age: {patient.age} ({patient.age_group})
- Weight: {patient.weight_kg} kg
- Assigned ESI: {esi_score if esi_score is not None else 'not assigned'}
- Chief Complaint: {patient.chief_complaint}

VITALS:
- HR: {patient.vitals.heart_rate} bpm
- BP: {patient.vitals.bp_display}
- RR: {patient.vitals.respiratory_rate} breaths/min
- SpO2: {patient.vitals.spo2}%
- Temp: {patient.vitals.temperature_c}°C ({patient.vitals.temperature_f}°F)

ESCALATION TRIGGERS:
{chr(10).join(f'- {r}' for r in reasons) if reasons else '- None recorded'}

ABNORMAL VITALS DETECTED:
{chr(10).join(f'- {v}' for v in abnormal_vitals) if abnormal_vitals else '- None detected by automated check'}

Generate the physician summary now.
"""

    try:
        # Construct inside the try — an auth or config failure here must fall
        # through to the static summary below, not propagate.
        llm = _get_llm().with_structured_output(_SummaryContent)
        summary: _SummaryContent = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=alert_input),
        ])
        return PhysicianSummary(**summary.model_dump())

    except Exception as e:
        logger.error("Physician summary generation failed for %s: %s", patient.patient_id, e)
        # The escalation itself still stands — only the narrative is missing.
        return PhysicianSummary(
            urgency_statement=f"{patient.patient_id} requires immediate physician evaluation.",
            trigger_reasons=reasons or ["Escalation triggered; detail unavailable."],
            abnormal_vitals=abnormal_vitals,
            clinical_concerns=(
                "Automated summary generation failed. The escalation decision itself is valid — "
                "review the trigger reasons and the patient directly."
            ),
            recommended_actions=["Immediate physician bedside evaluation required."],
        )
