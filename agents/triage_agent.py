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

from config import ANTHROPIC_API_KEY, LLM_MODEL
from models import (
    ClinicalFinding,
    Patient,
    PatientCard,
    TriageResult,
    TriageState,
    TriageStatus,
)
from agents.assessment import assess_patient, format_thresholds_for_prompt
from agents.escalation import build_escalation
from rag.retriever import retrieve_triage_context

logger = logging.getLogger(__name__)


# ─── Internal LLM Output Schema ──────────────────────────────────────────────

class _TriageDecision(BaseModel):
    """Structured output schema for the triage reasoning node."""
    esi_score: int                       = Field(ge=1, le=5, description="Final ESI score 1–5")
    esi_rationale: str                   = Field(description="Why this ESI level was assigned per ESI v4 criteria")
    clinical_reasoning: str              = Field(description="Full clinical reasoning narrative for this patient")
    recommended_interventions: list[str] = Field(description="Immediate nursing or clinical interventions to initiate")


# ─── LLM Initializer ─────────────────────────────────────────────────────────

def _get_llm() -> ChatAnthropic:
    """Return a configured Claude instance."""
    return ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )


# ─── Node 1 — Deterministic Assessment ───────────────────────────────────────

def assess_node(state: TriageState) -> TriageState:
    """
    Run the rule-based clinical checks. No LLM call, no API cost.

    Runs first so the findings can be handed to the triage LLM as context.
    """
    patient: Patient = state["patient"]
    logger.info("assess_node — %s", patient.patient_id)

    return {**state, "findings": assess_patient(patient)}


# ─── Node 2 — Triage Reasoning ───────────────────────────────────────────────

def _format_findings(findings: list[ClinicalFinding]) -> str:
    if not findings:
        return "None. All vitals within range and no keyword-matched symptoms."
    return "\n".join(f"- [{f.severity.value}] {f.detail}" for f in findings)


def triage_node(state: TriageState) -> TriageState:
    """
    Assign an ESI score, grounded in retrieved guidelines and informed by the
    deterministic findings from assess_node.

    Every patient reaches this node. There is no path that skips scoring.
    """
    patient: Patient = state["patient"]
    findings: list[ClinicalFinding] = state.get("findings", [])
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

The automated findings below come from a keyword and threshold scan. Treat them
as a checklist, not a conclusion. In particular, the symptom scan cannot detect
negation or history, so a finding of "chest pain" may reflect a complaint of
"denies chest pain" or "history of chest pain". Read the chief complaint yourself
and disregard any finding the text does not actually support.

Assign the ESI score that most accurately reflects the patient's acuity.
Recommend specific, actionable nursing interventions appropriate to the ESI level."""

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

AUTOMATED FINDINGS:
{_format_findings(findings)}

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
        )
        logger.info("triage_node — ESI %d assigned to %s", decision.esi_score, patient.patient_id)
        return {**state, "triage_result": triage_result, "retrieval_context": rag_context}

    except Exception as e:
        # No score is better than a fabricated one. escalate_node turns a missing
        # score into an IMMEDIATE escalation flagged as a system error, so the
        # patient surfaces for manual triage rather than sitting on a guess.
        logger.error("triage_node LLM call failed for %s: %s", patient.patient_id, e)
        return {
            **state,
            "triage_result": None,
            "retrieval_context": rag_context,
            "system_error": f"{type(e).__name__}: {e}",
        }


# ─── Node 3 — Escalation ─────────────────────────────────────────────────────

def escalate_node(state: TriageState) -> TriageState:
    """
    Derive the escalation level from the ESI score and the deterministic
    findings, and generate a physician summary when it is IMMEDIATE.
    """
    patient: Patient = state["patient"]
    triage_result: TriageResult | None = state.get("triage_result")
    findings: list[ClinicalFinding] = state.get("findings", [])

    escalation = build_escalation(
        patient=patient,
        esi_score=triage_result.esi_score if triage_result else None,
        findings=findings,
        system_error=state.get("system_error"),
    )

    patient.status = (
        TriageStatus.ESCALATED if escalation.triggered else TriageStatus.TRIAGED
    )

    return {**state, "patient": patient, "escalation": escalation}


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
        "findings": [],
        "triage_result": None,
        "escalation": None,
        "patient_card": None,
        "retrieval_context": "",
        "system_error": None,
    }

    return graph.invoke(initial_state)["patient_card"]
