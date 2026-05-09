"""
agents/triage_agent.py — LangGraph StateGraph orchestrating the full triage pipeline.

Graph flow:
    START
      └─► red_flag_node
            ├─► (if flagged)  build_card_node ──► END
            └─► (if clear)    triage_node ──► build_card_node ──► END
"""

import logging
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from config import ANTHROPIC_API_KEY, LLM_MODEL
from models import Patient, TriageResult, PatientCard, TriageStatus, TriageState
from agents.red_flag import evaluate_red_flag
from rag.retriever import retrieve_triage_context

logger = logging.getLogger(__name__)


# ─── Internal LLM Output Schema ──────────────────────────────────────────────

class _TriageDecision(BaseModel):
    """Structured output schema for the standard triage LLM reasoning node."""
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


# ─── Node 1 — Red Flag Evaluation ────────────────────────────────────────────

def red_flag_node(state: TriageState) -> TriageState:
    """
    Run the three-layer red flag evaluation.
    Writes red_flag_alert and esi_estimate back to state.
    If flagged, updates patient status to RED_FLAGGED.
    """
    patient: Patient = state["patient"]
    logger.info("red_flag_node — evaluating %s", patient.patient_id)

    alert, esi_estimate = evaluate_red_flag(patient)

    if alert.triggered:
        patient.status = TriageStatus.RED_FLAGGED

    return {
        **state,
        "patient": patient,
        "red_flag_alert": alert,
        "retrieval_context": "",
        "esi_estimate": esi_estimate,
    }


# ─── Node 2 — Standard Triage Reasoning ──────────────────────────────────────

def triage_node(state: TriageState) -> TriageState:
    """
    LLM triage reasoning for patients who cleared the red flag gate.
    Pulls RAG context, assigns final ESI score, and generates clinical reasoning.
    """
    patient: Patient = state["patient"]
    esi_hint = str(state.get("esi_estimate", "")) if state.get("esi_estimate") else ""
    logger.info("triage_node — reasoning for %s", patient.patient_id)

    rag_context = retrieve_triage_context(patient.chief_complaint, esi_hint)

    llm = _get_llm().with_structured_output(_TriageDecision)

    system_prompt = """You are an experienced emergency triage nurse applying the ESI (Emergency Severity Index) v4 framework.

Your task is to assign a final ESI score and provide structured clinical reasoning for the patient.

ESI SCORING GUIDE:
- ESI 1: Requires immediate life-saving intervention
- ESI 2: High risk — should not wait, confused/lethargic/disoriented, or severe pain/distress
- ESI 3: Stable but requires 2+ resources (labs, IV, imaging, consults)
- ESI 4: Stable, requires exactly 1 resource
- ESI 5: Stable, no resources needed — can be seen and discharged

VITAL SIGN DANGER ZONES (auto-escalate ESI if present):
- HR > 100 or < 60
- RR > 20
- SpO2 < 94%
- SBP < 90 or > 180
- Temp > 38.5°C or < 36°C

Assign the ESI score that most accurately reflects the patient's acuity.
Recommend specific, actionable nursing interventions appropriate to the ESI level."""

    patient_data = f"""
PATIENT:
- Name: {patient.name}
- Age: {patient.age} ({patient.age_group})
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
        decision: _TriageDecision = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=patient_data),
        ])

        triage_result = TriageResult(
            esi_score=decision.esi_score,
            esi_rationale=decision.esi_rationale,
            clinical_reasoning=decision.clinical_reasoning,
            recommended_interventions=decision.recommended_interventions,
            retrieval_context_used=bool(rag_context),
        )

        patient.status = TriageStatus.TRIAGED
        logger.info("triage_node — ESI %d assigned to %s", decision.esi_score, patient.patient_id)

    except Exception as e:
        logger.error("triage_node LLM call failed for %s: %s", patient.patient_id, e)
        triage_result = TriageResult(
            esi_score=3,
            esi_rationale="Automated triage failed — defaulting to ESI 3 pending manual review.",
            clinical_reasoning=str(e),
            recommended_interventions=["Manual triage review required."],
            retrieval_context_used=False,
        )

    return {
        **state,
        "patient": patient,
        "triage_result": triage_result,
        "retrieval_context": rag_context,
    }


# ─── Node 3 — Patient Card Builder ───────────────────────────────────────────

def build_card_node(state: TriageState) -> TriageState:
    """
    Assemble the final PatientCard from state.
    This is the structured output rendered by the Streamlit UI.
    """
    patient: Patient           = state["patient"]
    red_flag_alert             = state["red_flag_alert"]
    triage_result              = state.get("triage_result")

    card = PatientCard(
        patient=patient,
        red_flag_alert=red_flag_alert,
        triage_result=triage_result,
    )

    logger.info("build_card_node — card built for %s | %s", patient.patient_id, card.display_esi)

    return {**state, "patient_card": card}


# ─── Conditional Routing ──────────────────────────────────────────────────────

def _route_after_red_flag(state: TriageState) -> str:
    """Route to triage_node if clear, skip straight to build_card_node if flagged."""
    if state["red_flag_alert"].triggered:
        return "build_card_node"
    return "triage_node"


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_triage_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph triage pipeline.
    Returns a compiled graph ready to invoke with a TriageState dict.
    """
    graph = StateGraph(TriageState)

    graph.add_node("red_flag_node", red_flag_node)
    graph.add_node("triage_node", triage_node)
    graph.add_node("build_card_node", build_card_node)

    graph.add_edge(START, "red_flag_node")
    graph.add_conditional_edges(
        "red_flag_node",
        _route_after_red_flag,
        {
            "triage_node": "triage_node",
            "build_card_node": "build_card_node",
        },
    )
    graph.add_edge("triage_node", "build_card_node")
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
        "red_flag_alert": None,
        "triage_result": None,
        "patient_card": None,
        "retrieval_context": "",
        "esi_estimate": 0,
        "error": None,
    }

    final_state = graph.invoke(initial_state)
    return final_state["patient_card"]
