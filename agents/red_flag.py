"""
agents/red_flag.py — Three-layer red flag evaluation system.

Layer 1: Hard ESI rule   — embedded inside Layer 3 LLM assessment.
Layer 2: Deterministic   — vital sign danger zones, symptom keywords, age-risk amplifier.
Layer 3: LLM reasoning   — holistic clinical picture grounded in RAG context.

Any layer triggering generates a structured PhysicianSummary and returns a RedFlagAlert.
"""

import logging
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    RED_FLAG_SYMPTOMS,
    VITAL_THRESHOLDS,
    AGE_THRESHOLDS,
    ESI_RED_FLAG_MAX,
)
from models import Patient, VitalSigns, RedFlagAlert, RedFlagLayer, PhysicianSummary
from rag.retriever import retrieve_red_flag_context

logger = logging.getLogger(__name__)


# ─── Internal LLM Output Schemas ─────────────────────────────────────────────

class _RedFlagDecision(BaseModel):
    """Structured output schema for the Layer 3 red flag LLM call."""
    is_red_flag: bool      = Field(description="True if the patient requires immediate physician attention")
    esi_estimate: int      = Field(ge=1, le=5, description="Estimated ESI level 1–5 based on full clinical picture")
    primary_concerns: list[str] = Field(description="Specific clinical concerns driving this decision")
    reasoning: str         = Field(description="Clinical reasoning narrative — how you arrived at this decision")


class _SummaryContent(BaseModel):
    """Structured output schema for the physician summary LLM call."""
    urgency_statement: str          = Field(description="One-line urgency callout for the physician")
    trigger_reasons: list[str]      = Field(description="Exactly why the red flag fired — bullet points")
    abnormal_vitals: list[str]      = Field(description="Out-of-range vitals with measured values")
    clinical_concerns: str          = Field(description="Paragraph summarizing the primary clinical concerns")
    recommended_actions: list[str]  = Field(description="Immediate actions the physician should consider")


# ─── LLM Initializer ─────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    """Return a configured Claude instance."""
    return ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,  # deterministic — clinical decisions should not vary
    )


# ─── Layer 2 — Deterministic Checks ──────────────────────────────────────────

def _check_vital_signs(vitals: VitalSigns) -> list[str]:
    """
    Compare patient vitals against ESI v4 danger zone thresholds.
    Returns a list of human-readable abnormal findings.
    """
    findings: list[str] = []

    if vitals.heart_rate > VITAL_THRESHOLDS["hr_high"]:
        findings.append(f"Tachycardia — HR {vitals.heart_rate} bpm (threshold >{VITAL_THRESHOLDS['hr_high']})")
    if vitals.heart_rate < VITAL_THRESHOLDS["hr_low"]:
        findings.append(f"Bradycardia — HR {vitals.heart_rate} bpm (threshold <{VITAL_THRESHOLDS['hr_low']})")
    if vitals.respiratory_rate > VITAL_THRESHOLDS["rr_high"]:
        findings.append(f"Tachypnea — RR {vitals.respiratory_rate} breaths/min (threshold >{VITAL_THRESHOLDS['rr_high']})")
    if vitals.spo2 < VITAL_THRESHOLDS["spo2_low"]:
        findings.append(f"Hypoxia — SpO2 {vitals.spo2}% (threshold <{VITAL_THRESHOLDS['spo2_low']}%)")
    if vitals.temperature_c > VITAL_THRESHOLDS["temp_high_c"]:
        findings.append(f"Fever — Temp {vitals.temperature_c}°C / {vitals.temperature_f}°F (threshold >{VITAL_THRESHOLDS['temp_high_c']}°C)")
    if vitals.temperature_c < VITAL_THRESHOLDS["temp_low_c"]:
        findings.append(f"Hypothermia — Temp {vitals.temperature_c}°C / {vitals.temperature_f}°F (threshold <{VITAL_THRESHOLDS['temp_low_c']}°C)")
    if vitals.systolic_bp > VITAL_THRESHOLDS["sbp_high"]:
        findings.append(f"Hypertensive urgency — SBP {vitals.systolic_bp} mmHg (threshold >{VITAL_THRESHOLDS['sbp_high']})")
    if vitals.systolic_bp < VITAL_THRESHOLDS["sbp_low"]:
        findings.append(f"Hypotension — SBP {vitals.systolic_bp} mmHg (threshold <{VITAL_THRESHOLDS['sbp_low']})")

    return findings


def _check_symptoms(chief_complaint: str) -> list[str]:
    """
    Scan chief complaint text against the red flag symptom keyword list.
    Returns matched symptom strings.
    """
    complaint_lower = chief_complaint.lower()
    return [
        f"Red flag symptom identified: '{symptom}'"
        for symptom in RED_FLAG_SYMPTOMS
        if symptom in complaint_lower
    ]


def _run_layer_2(patient: Patient) -> tuple[bool, list[str], list[str]]:
    """
    Full Layer 2 evaluation — vitals, symptoms, and age-risk amplifier.

    Age risk logic: a high-risk age group (pediatric or geriatric) combined
    with any other abnormal finding elevates the flag. Age alone does not trigger —
    that nuance is handled by the Layer 3 LLM.

    Returns:
        triggered       — True if Layer 2 fires
        all_reasons     — full list of findings for the physician summary
        abnormal_vitals — vital findings only, used separately in the summary
    """
    abnormal_vitals = _check_vital_signs(patient.vitals)
    triggered_symptoms = _check_symptoms(patient.chief_complaint)

    all_reasons: list[str] = []
    all_reasons.extend(abnormal_vitals)
    all_reasons.extend(triggered_symptoms)

    # Age amplifier — high-risk demographic with any concurrent concerning finding
    if patient.age_group in ("pediatric", "geriatric") and all_reasons:
        all_reasons.append(
            f"High-risk age group: {patient.age_group} patient (age {patient.age}) "
            f"with concurrent abnormal finding — elevated clinical concern"
        )

    triggered = bool(all_reasons)
    return triggered, all_reasons, abnormal_vitals


# ─── Layer 3 — LLM Holistic Reasoning ────────────────────────────────────────

def _run_layer_3(patient: Patient, rag_context: str) -> tuple[bool, list[str], int]:
    """
    LLM evaluates the full patient picture against clinical guidelines.
    Also performs the Layer 1 ESI check — if the estimated ESI is 1 or 2, flag fires.

    Returns:
        triggered      — True if LLM flags the patient
        reasons        — LLM-identified clinical concerns
        esi_estimate   — LLM's preliminary ESI score (passed to triage node if not flagged)
    """
    llm = _get_llm().with_structured_output(_RedFlagDecision)

    system_prompt = """You are a senior emergency physician performing a rapid triage risk assessment.

Your task is to evaluate whether this patient requires IMMEDIATE physician attention (red flag).

Apply the ESI (Emergency Severity Index) framework:
- ESI 1: Immediate life threat — requires immediate intervention
- ESI 2: Emergent — high risk, should not wait
- ESI 3: Urgent — stable but needs multiple resources
- ESI 4: Less urgent — one resource needed
- ESI 5: Non-urgent — no resources needed

RED FLAG if ANY of the following:
- Estimated ESI level 1 or 2
- Any combination of age, weight, vitals, or complaint suggesting high-acuity or rapid deterioration
- Atypical presentations in elderly or pediatric patients that mask serious pathology
- Vital signs trending toward instability even if currently borderline

Use the clinical reference context to support your reasoning.
Be conservative — it is safer to over-triage than to miss a critical patient."""

    patient_summary = f"""
PATIENT PRESENTATION:
- Name: {patient.name}
- Age: {patient.age} years ({patient.age_group})
- Weight: {patient.weight_kg} kg ({patient.weight_lbs} lbs)
- Chief Complaint: {patient.chief_complaint}

VITAL SIGNS:
- Heart Rate: {patient.vitals.heart_rate} bpm
- Blood Pressure: {patient.vitals.bp_display}
- Respiratory Rate: {patient.vitals.respiratory_rate} breaths/min
- SpO2: {patient.vitals.spo2}%
- Temperature: {patient.vitals.temperature_c}°C ({patient.vitals.temperature_f}°F)

CLINICAL REFERENCE CONTEXT:
{rag_context}
"""

    try:
        decision: _RedFlagDecision = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=patient_summary),
        ])

        # Layer 1 embedded — ESI 1 or 2 always triggers regardless of LLM flag
        if decision.esi_estimate <= ESI_RED_FLAG_MAX:
            decision.is_red_flag = True
            decision.primary_concerns.insert(
                0, f"ESI estimate {decision.esi_estimate} — meets Level 1/2 red flag threshold"
            )

        logger.info(
            "Layer 3 — is_red_flag=%s, ESI estimate=%d, patient=%s",
            decision.is_red_flag, decision.esi_estimate, patient.patient_id,
        )
        return decision.is_red_flag, decision.primary_concerns, decision.esi_estimate

    except Exception as e:
        logger.error("Layer 3 LLM call failed for %s: %s", patient.patient_id, e)
        # Fail safe — flag the patient if the LLM errors
        return True, ["Layer 3 evaluation error — flagging for physician safety review"], 0


# ─── Physician Summary Generator ─────────────────────────────────────────────

def _generate_physician_summary(
    patient: Patient,
    layer: RedFlagLayer,
    reasons: list[str],
    abnormal_vitals: list[str],
) -> PhysicianSummary:
    """
    Generate a structured physician alert summary via LLM.
    Called only when a red flag has been confirmed by any layer.
    """
    llm = _get_llm().with_structured_output(_SummaryContent)

    system_prompt = """You are generating a concise, actionable alert summary for an emergency physician.
The patient has been red-flagged by the triage system and requires immediate attention.

Write with clinical precision. Be direct. Avoid filler language.
Recommended actions should be specific and immediately actionable."""

    alert_input = f"""
RED FLAG TRIGGERED — {layer.value}

PATIENT:
- Name: {patient.name}
- Age: {patient.age} ({patient.age_group})
- Weight: {patient.weight_kg} kg ({patient.weight_lbs} lbs)
- Chief Complaint: {patient.chief_complaint}

VITALS:
- HR: {patient.vitals.heart_rate} bpm
- BP: {patient.vitals.bp_display}
- RR: {patient.vitals.respiratory_rate} breaths/min
- SpO2: {patient.vitals.spo2}%
- Temp: {patient.vitals.temperature_c}°C ({patient.vitals.temperature_f}°F)

TRIGGER REASONS:
{chr(10).join(f'- {r}' for r in reasons)}

ABNORMAL VITALS DETECTED:
{chr(10).join(f'- {v}' for v in abnormal_vitals) if abnormal_vitals else 'None detected by automated check'}

Generate the physician summary now.
"""

    try:
        summary: _SummaryContent = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=alert_input),
        ])
        return PhysicianSummary(
            urgency_statement=summary.urgency_statement,
            trigger_reasons=summary.trigger_reasons,
            abnormal_vitals=summary.abnormal_vitals,
            clinical_concerns=summary.clinical_concerns,
            recommended_actions=summary.recommended_actions,
        )
    except Exception as e:
        logger.error("Physician summary generation failed for %s: %s", patient.patient_id, e)
        return PhysicianSummary(
            urgency_statement=f"ALERT — {patient.name} requires immediate physician evaluation.",
            trigger_reasons=reasons,
            abnormal_vitals=abnormal_vitals,
            clinical_concerns="Automated summary generation failed. Review patient directly.",
            recommended_actions=["Immediate physician bedside evaluation required."],
        )


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def evaluate_red_flag(patient: Patient) -> tuple[RedFlagAlert, int]:
    """
    Run all three red flag layers against the patient record.

    Evaluation order:
        Layer 2 first (deterministic, no LLM cost)
        Layer 3 only if Layer 2 does not trigger (LLM + RAG)

    Returns:
        RedFlagAlert — full alert object with physician summary if triggered
        esi_estimate — preliminary ESI from Layer 3 (0 if Layer 2 caught it first)
    """
    logger.info("Starting red flag evaluation for patient %s", patient.patient_id)

    # ── Layer 2 — Deterministic ───────────────────────────────────────────────
    l2_triggered, l2_reasons, l2_abnormal_vitals = _run_layer_2(patient)

    if l2_triggered:
        logger.info("Layer 2 triggered for %s — skipping Layer 3", patient.patient_id)
        summary = _generate_physician_summary(
            patient, RedFlagLayer.LAYER_2_CLINICAL, l2_reasons, l2_abnormal_vitals
        )
        return RedFlagAlert(
            triggered=True,
            layer=RedFlagLayer.LAYER_2_CLINICAL,
            reasons=l2_reasons,
            physician_summary=summary,
        ), 0

    # ── Layer 3 — LLM Holistic Reasoning (includes Layer 1 ESI check) ────────
    rag_context = retrieve_red_flag_context(patient.age, patient.chief_complaint)
    l3_triggered, l3_reasons, esi_estimate = _run_layer_3(patient, rag_context)

    if l3_triggered:
        layer = (
            RedFlagLayer.LAYER_1_ESI
            if esi_estimate > 0 and esi_estimate <= ESI_RED_FLAG_MAX
            else RedFlagLayer.LAYER_3_LLM
        )
        summary = _generate_physician_summary(
            patient, layer, l3_reasons, l2_abnormal_vitals
        )
        return RedFlagAlert(
            triggered=True,
            layer=layer,
            reasons=l3_reasons,
            physician_summary=summary,
        ), esi_estimate

    # ── No Flag ───────────────────────────────────────────────────────────────
    logger.info("No red flag triggered for %s — passing to triage", patient.patient_id)
    return RedFlagAlert(triggered=False), esi_estimate
