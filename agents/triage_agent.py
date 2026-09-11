"""
agents/triage_agent.py — LangGraph StateGraph orchestrating the full triage pipeline.

Graph flow (linear — every patient traverses every node):

    START ─► assess_node ─► triage_node ─► escalate_node ─► build_card_node ─► END

Why linear: acuity scoring and escalation are one clinical judgment, not two
competing branches. Every patient gets an ESI score. Escalation is derived from
that score plus the deterministic findings, and is reported alongside the score
rather than instead of it.
"""

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from config import ANTHROPIC_API_KEY, TRIAGE_MODEL
from models import (
    ClinicalFinding,
    ExtractedSymptom,
    Patient,
    PatientCard,
    TriageResult,
    TriageState,
    TriageStatus,
)
from agents.assessment import (
    assess_symptoms,
    assess_vitals,
    combine_findings,
    findings_from_extraction,
    format_symptom_vocabulary_for_prompt,
    format_thresholds_for_prompt,
)
from agents.escalation import build_escalation
from rag.retriever import retrieve_triage_context

logger = logging.getLogger(__name__)


# ─── Internal LLM Output Schema ──────────────────────────────────────────────

class _TriageDecision(BaseModel):
    """
    Structured output schema for the triage reasoning node.

    Symptom extraction rides along on this call rather than taking one of its
    own: the model is already reading the chief complaint to assign a score, so
    reporting what it found there costs nothing extra.
    """
    extracted_symptoms: list[ExtractedSymptom] = Field(
        description="Vocabulary symptoms the complaint mentions, each marked present, denied, or historical"
    )
    esi_score: int                       = Field(ge=1, le=5, description="Final ESI score 1–5")
    esi_rationale: str                   = Field(description="Why this ESI level was assigned per ESI v4 criteria")
    clinical_reasoning: str              = Field(description="Full clinical reasoning narrative for this patient")
    recommended_interventions: list[str] = Field(description="Immediate nursing or clinical interventions to initiate")


# ─── LLM Initializer ─────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    """Return the Claude instance used for ESI scoring."""
    return ChatAnthropic(
        model=TRIAGE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )


# ─── Node 1 — Deterministic Assessment ───────────────────────────────────────

def assess_node(state: TriageState) -> TriageState:
    """
    Check the vital signs against the threshold tiers. No LLM call, no API cost.

    Vitals only. Symptoms come from structured extraction in triage_node, and
    age risk is applied in escalate_node once both are known. Thresholds stay
    deterministic here on purpose: a number against a number is not a judgment
    call, and it should not be delegated to a model.
    """
    patient: Patient = state["patient"]
    logger.info("assess_node — %s", patient.patient_id)

    return {**state, "vital_findings": assess_vitals(patient.vitals)}


# ─── Node 2 — Triage Reasoning ───────────────────────────────────────────────

def _format_findings(findings: list[ClinicalFinding]) -> str:
    if not findings:
        return "None. All vital signs within range."
    return "\n".join(f"- [{f.severity.value}] {f.detail}" for f in findings)


def triage_node(state: TriageState) -> TriageState:
    """
    Assign an ESI score and extract symptom status from the chief complaint.

    Two jobs, one call: the model has to read the complaint to score it anyway,
    so it also reports which vocabulary symptoms the text asserts, denies, or
    places in the past.

    Every patient reaches this node. There is no path that skips scoring.
    """
    patient: Patient = state["patient"]
    vital_findings: list[ClinicalFinding] = state.get("vital_findings", [])
    logger.info("triage_node — reasoning for %s", patient.patient_id)

    rag_context, context_found = retrieve_triage_context(patient.chief_complaint)

    system_prompt = f"""You are an experienced emergency triage nurse applying the ESI (Emergency Severity Index) v4 framework.

Assign a final ESI score and provide structured clinical reasoning.

ESI SCORING GUIDE:
- ESI 1: Requires immediate life-saving intervention
- ESI 2: High risk — should not wait, confused/lethargic/disoriented, or severe pain/distress
- ESI 3: Stable but requires 2+ resources (labs, IV, imaging, consults)
- ESI 4: Stable, requires exactly 1 resource
- ESI 5: Stable, no resources needed — can be seen and discharged

VITAL SIGN THRESHOLDS (adult ranges):
{format_thresholds_for_prompt()}

The vital sign findings below are measured values compared against those
thresholds. They are reliable — treat them as fact.

SYMPTOM EXTRACTION

Separately, report the status of any symptom from this vocabulary that the
chief complaint mentions:

{format_symptom_vocabulary_for_prompt()}

For each vocabulary term the complaint refers to in any form, return one of:
- "present"    — the patient has this now
- "denied"     — the complaint explicitly says the patient does NOT have it
                 ("denies chest pain", "no shortness of breath")
- "historical" — it refers to a past episode, not this presentation
                 ("history of stroke in 2019")

Rules:
- Use the vocabulary terms exactly as written above. Do not invent new terms.
- Omit any term the complaint does not refer to at all. An empty list is correct
  for a complaint that mentions none of them.
- Status reflects only what the text says. Do not infer a symptom the patient
  did not report, and do not upgrade a denial into a presence because the vital
  signs look concerning.

Then assign the ESI score that most accurately reflects the patient's acuity,
and recommend specific, actionable nursing interventions appropriate to that
level."""

    patient_data = f"""
PATIENT:
- ID: {patient.patient_id}
- Age: {patient.age} ({patient.age_group})
- Weight: {patient.weight_kg} kg
- Chief Complaint: {patient.chief_complaint}

VITAL SIGNS:
- Heart Rate: {patient.vitals.heart_rate} bpm
- Blood Pressure: {patient.vitals.bp_display}
- Respiratory Rate: {patient.vitals.respiratory_rate} breaths/min
- SpO2: {patient.vitals.spo2}%
- Temperature: {patient.vitals.temperature_c}°C ({patient.vitals.temperature_f}°F)

MEASURED VITAL SIGN FINDINGS:
{_format_findings(vital_findings)}

CLINICAL REFERENCE CONTEXT:
{rag_context}
"""

    try:
        # Client construction is inside the try: a missing or malformed API key
        # fails here, not at invoke, and must reach the fail-safe path below
        # rather than crashing the graph.
        llm = _get_llm().with_structured_output(_TriageDecision)
        decision: _TriageDecision = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=patient_data),
        ])

        triage_result = TriageResult(
            esi_score=decision.esi_score,
            esi_rationale=decision.esi_rationale,
            clinical_reasoning=decision.clinical_reasoning,
            recommended_interventions=decision.recommended_interventions,
            retrieval_context_used=context_found,
            extracted_symptoms=decision.extracted_symptoms,
        )
        logger.info(
            "triage_node — ESI %d assigned to %s (%d symptom(s) extracted)",
            decision.esi_score, patient.patient_id, len(decision.extracted_symptoms),
        )
        return {
            **state,
            "triage_result": triage_result,
            "symptom_findings": findings_from_extraction(decision.extracted_symptoms),
            "retrieval_context": rag_context,
        }

    except Exception as e:
        # No score is better than a fabricated one. escalate_node turns a missing
        # score into an IMMEDIATE escalation flagged as a system error, so the
        # patient surfaces for manual triage rather than sitting on a guess.
        logger.error("triage_node LLM call failed for %s: %s", patient.patient_id, e)
        return {
            **state,
            "triage_result": None,
            # Fall back to the naive substring scan. It cannot handle negation,
            # but losing symptom detection entirely on an already-degraded path
            # would be worse than a few false positives.
            "symptom_findings": assess_symptoms(patient.chief_complaint),
            "retrieval_context": rag_context,
            "system_error": f"{type(e).__name__}: {e}",
        }


# ─── Node 3 — Escalation ─────────────────────────────────────────────────────

def escalate_node(state: TriageState) -> TriageState:
    """
    Combine the measured vital findings with the extracted symptom findings,
    apply the age amplifier to the result, then derive the escalation level and
    generate a physician summary when it is IMMEDIATE.

    Age risk is applied here rather than in assess_node because it depends on
    whether any other finding exists, and symptom findings are not known until
    triage_node has extracted them.
    """
    patient: Patient = state["patient"]
    triage_result: TriageResult | None = state.get("triage_result")

    findings = combine_findings(
        patient,
        state.get("vital_findings", []),
        state.get("symptom_findings", []),
    )

    escalation = build_escalation(
        patient=patient,
        esi_score=triage_result.esi_score if triage_result else None,
        findings=findings,
        system_error=state.get("system_error"),
    )

    patient.status = (
        TriageStatus.ESCALATED if escalation.triggered else TriageStatus.TRIAGED
    )

    return {**state, "patient": patient, "findings": findings, "escalation": escalation}


# ─── Node 4 — Patient Card Builder ───────────────────────────────────────────

def build_card_node(state: TriageState) -> TriageState:
    """Assemble the final PatientCard rendered by the Streamlit UI."""
    patient: Patient = state["patient"]

    card = PatientCard(
        patient=patient,
        escalation=state["escalation"],
        triage_result=state.get("triage_result"),
    )

    logger.info(
        "build_card_node — %s | %s | escalation=%s",
        patient.patient_id, card.display_esi, card.escalation.level.value,
    )
    return {**state, "patient_card": card}


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_triage_graph():
    """Assemble and compile the LangGraph triage pipeline."""
    graph = StateGraph(TriageState)

    graph.add_node("assess_node", assess_node)
    graph.add_node("triage_node", triage_node)
    graph.add_node("escalate_node", escalate_node)
    graph.add_node("build_card_node", build_card_node)

    graph.add_edge(START, "assess_node")
    graph.add_edge("assess_node", "triage_node")
    graph.add_edge("triage_node", "escalate_node")
    graph.add_edge("escalate_node", "build_card_node")
    graph.add_edge("build_card_node", END)

    return graph.compile()


# ─── Public Interface ─────────────────────────────────────────────────────────

def run_triage(patient: Patient) -> PatientCard:
    """
    Run a single patient through the full triage pipeline.
    Returns the completed PatientCard for display and storage.
    """
    graph = build_triage_graph()

    initial_state: TriageState = {
        "patient": patient,
        "vital_findings": [],
        "symptom_findings": [],
        "findings": [],
        "triage_result": None,
        "escalation": None,
        "patient_card": None,
        "retrieval_context": "",
        "system_error": None,
    }

    return graph.invoke(initial_state)["patient_card"]
