"""
main.py — Streamlit UI for the ED Triage System.
"""

import logging

import streamlit as st

from rag.ingest import ingest
from agents.assessment import vital_severity_map
from agents.triage_agent import run_triage
from memory.patient_store import store
from models import (
    EscalationLevel,
    FindingSeverity,
    Patient,
    PatientCard,
    TriageStatus,
    VitalSigns,
)

logger = logging.getLogger(__name__)

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="ED Triage System",
    layout="wide",
    # "auto" collapses the sidebar on small screens. Pinned open, the intake
    # form covered the entire queue on a phone.
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
.escalation-banner {
    color: white; padding: 10px 16px; border-radius: 6px;
    font-size: 1.15rem; font-weight: bold; margin-bottom: 10px;
}
.escalation-immediate { background-color: #c0392b; }
.escalation-elevated  { background-color: #e67e22; }
.escalation-error     { background-color: #7f8c8d; }
.esi-badge {
    display: inline-block;
    padding: 5px 14px; border-radius: 5px;
    font-weight: bold; font-size: 1rem; color: white;
}
.vital-ok         { color: #27ae60; font-weight: 500; }
.vital-concerning { color: #c07a00; font-weight: bold; }
.vital-critical   { color: #c0392b; font-weight: bold; }
.returning-tag {
    background: #2980b9; color: white;
    font-size: 0.75rem; padding: 2px 8px;
    border-radius: 4px; margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── ESI Display Helpers ──────────────────────────────────────────────────────

_ESI_COLORS = {1: "#c0392b", 2: "#e67e22", 3: "#f1c40f", 4: "#27ae60", 5: "#2980b9"}
_ESI_LABELS = {
    1: "ESI 1 — Immediate",
    2: "ESI 2 — Emergent",
    3: "ESI 3 — Urgent",
    4: "ESI 4 — Less Urgent",
    5: "ESI 5 — Non-Urgent",
}

# Escalation is shown alongside the ESI score, never instead of it.
_ESCALATION_LABELS = {
    EscalationLevel.IMMEDIATE: "PHYSICIAN NOW",
    EscalationLevel.ELEVATED: "ELEVATED CONCERN",
}
_ESCALATION_CSS = {
    EscalationLevel.IMMEDIATE: "escalation-immediate",
    EscalationLevel.ELEVATED: "escalation-elevated",
}

# Severity is carried by a text marker as well as colour, so the signal
# survives for colourblind users and in greyscale.
_SEVERITY_STYLE = {
    FindingSeverity.CRITICAL: ("vital-critical", " ▲ CRITICAL"),
    FindingSeverity.CONCERNING: ("vital-concerning", " △ abnormal"),
}


def _esi_badge(score: int) -> str:
    """Return an HTML badge string for the given ESI score."""
    color = _ESI_COLORS.get(score, "#7f8c8d")
    label = _ESI_LABELS.get(score, f"ESI {score}")
    return f'<span class="esi-badge" style="background:{color};">{label}</span>'


def _escalation_banner(card: PatientCard) -> str | None:
    """Return the HTML escalation banner for a card, or None if routine."""
    if card.has_system_error:
        return (
            '<div class="escalation-banner escalation-error">'
            'SYSTEM ERROR — MANUAL TRIAGE REQUIRED</div>'
        )
    label = _ESCALATION_LABELS.get(card.escalation.level)
    if label is None:
        return None
    css = _ESCALATION_CSS[card.escalation.level]
    return f'<div class="escalation-banner {css}">{label}</div>'


# ─── RAG Initialization ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading clinical knowledge base...")
def _init_rag() -> bool:
    """
    Build or load the vector store once per session.

    Returns whether grounding is available. Failure here is not fatal: the app
    is designed to triage without retrieved context, so a missing knowledge
    base or a missing embedding dependency degrades rather than crashes.
    """
    try:
        return ingest() is not None
    except Exception as e:
        logger.warning("Knowledge base unavailable: %s", e)
        return False


# ─── Vitals Renderer ─────────────────────────────────────────────────────────

def _render_vitals(card: PatientCard) -> None:
    """
    Render vitals in two columns, marking out-of-range values.

    Severity comes from the same threshold logic the clinical rules use, so the
    display can never disagree with the assessment.
    """
    v = card.patient.vitals
    severities = vital_severity_map(v)

    def _span(text: str, *attrs: str) -> str:
        """Style a value by the worst severity across the attributes it shows."""
        found = [severities[a] for a in attrs if a in severities]
        if FindingSeverity.CRITICAL in found:
            css, marker = _SEVERITY_STYLE[FindingSeverity.CRITICAL]
        elif found:
            css, marker = _SEVERITY_STYLE[FindingSeverity.CONCERNING]
        else:
            css, marker = "vital-ok", ""
        return f'<span class="{css}">{text}{marker}</span>'

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Heart Rate:** {_span(f'{v.heart_rate} bpm', 'heart_rate')}",
            unsafe_allow_html=True,
        )
        # Both pressures are evaluated — a normal systolic must not mask a
        # dangerous diastolic.
        st.markdown(
            f"**Blood Pressure:** {_span(v.bp_display, 'systolic_bp', 'diastolic_bp')}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Resp Rate:** {_span(f'{v.respiratory_rate} breaths/min', 'respiratory_rate')}",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"**SpO2:** {_span(f'{v.spo2}%', 'spo2')}", unsafe_allow_html=True)
        st.markdown(
            f"**Temperature:** "
            f"{_span(f'{v.temperature_c}°C / {v.temperature_f}°F', 'temperature_c')}",
            unsafe_allow_html=True,
        )


# ─── Patient Card Renderer ────────────────────────────────────────────────────

def _render_patient_card(card: PatientCard) -> None:
    """Render the full structured patient card for the healthcare team."""
    p = card.patient

    # ── Header ────────────────────────────────────────────────────────────────
    col_hdr, col_badge = st.columns([3, 1])
    with col_hdr:
        returning_html = '<span class="returning-tag">RETURNING</span>' if p.is_returning else ""
        st.markdown(f"### {p.name} {returning_html}", unsafe_allow_html=True)
        st.caption(
            f"ID: **{p.patient_id}**  |  "
            f"Checked in: {p.check_in_time.strftime('%H:%M  —  %b %d, %Y')}"
        )
        st.markdown(
            f"**Age:** {p.age} yrs  |  "
            f"**Weight:** {p.weight_kg} kg ({p.weight_lbs} lbs)  |  "
            f"**Group:** {p.age_group.title()}"
        )

    with col_badge:
        # Score and escalation are shown together. An escalated patient still
        # has an ESI level, and hiding it was the original design's core error.
        if card.triage_result:
            st.markdown(_esi_badge(card.triage_result.esi_score), unsafe_allow_html=True)
        banner = _escalation_banner(card)
        if banner:
            st.markdown(banner, unsafe_allow_html=True)
        if not card.triage_result and not banner:
            st.info("Pending Triage")

    st.divider()

    # ── Chief Complaint ────────────────────────────────────────────────────────
    st.markdown(f"**Chief Complaint:** {p.chief_complaint}")
    st.markdown("**Vital Signs**")
    _render_vitals(card)

    st.divider()

    # ── System Error ───────────────────────────────────────────────────────────
    if card.has_system_error:
        st.error(
            "Automated triage did not complete for this patient, so no clinical "
            "judgment has been applied. Triage manually."
        )
        st.caption(f"Pipeline error: {card.escalation.system_error}")

    # ── Escalation ─────────────────────────────────────────────────────────────
    if card.is_escalated:
        summary = card.escalation.physician_summary

        if card.needs_immediate_attention:
            st.error("PHYSICIAN ALERT — immediate attention required")
        else:
            st.warning("Elevated concern — re-assess sooner than queue order suggests")

        if summary:
            st.markdown(f"**{summary.urgency_statement}**")

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown("**Trigger Reasons**")
                for r in summary.trigger_reasons:
                    st.markdown(f"- {r}")
                if summary.abnormal_vitals:
                    st.markdown("**Abnormal Vitals**")
                    for v in summary.abnormal_vitals:
                        st.markdown(f"- {v}")
            with col_r2:
                st.markdown("**Recommended Immediate Actions**")
                for action in summary.recommended_actions:
                    st.markdown(f"- {action}")

            st.markdown("**Clinical Concerns**")
            st.markdown(summary.clinical_concerns)
        else:
            # ELEVATED patients get no LLM summary, so show the raw triggers.
            st.markdown("**Why this patient was escalated**")
            for reason in card.escalation.reasons:
                st.markdown(f"- {reason}")

        st.divider()

    # ── Triage Result ──────────────────────────────────────────────────────────
    if card.triage_result:
        result = card.triage_result
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("**ESI Rationale**")
            st.markdown(result.esi_rationale)

        with col_t2:
            if result.recommended_interventions:
                st.markdown("**Recommended Interventions**")
                for item in result.recommended_interventions:
                    st.markdown(f"- {item}")

        with st.expander("Full Clinical Reasoning"):
            st.markdown(result.clinical_reasoning)
            if not result.retrieval_context_used:
                st.caption(
                    "Note: no clinical reference context was retrieved for this "
                    "patient. Reasoning relies on the model's training knowledge."
                )

    # ── Prior Visit History ────────────────────────────────────────────────────
    if p.is_returning:
        prior = [
            c for c in store.get_prior_visits(p.name)
            if c.patient.patient_id != p.patient_id
        ]
        if prior:
            with st.expander(f"Prior Visits ({len(prior)})"):
                for visit in prior:
                    st.markdown(
                        f"**{visit.patient.check_in_time.strftime('%b %d, %Y  %H:%M')}**  —  "
                        f"{visit.patient.chief_complaint}  —  {visit.display_esi}"
                    )


# ─── Queue Renderer ───────────────────────────────────────────────────────────

def _truncate(text: str, limit: int = 60) -> str:
    """Trim to a length with an explicit ellipsis so cut text is visible as cut."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _queue_status(card: PatientCard) -> str:
    """
    One-line status showing the score and the escalation together.

    Streamlit colour markup is paired with a text marker so the urgency is not
    conveyed by colour alone.
    """
    if card.has_system_error:
        return ":grey[**SYS ERROR**]"
    if card.triage_result is None:
        return "Pending"

    esi = f"ESI {card.triage_result.esi_score}"
    if card.needs_immediate_attention:
        return f":red[**{esi} · NOW**]"
    if card.is_escalated:
        return f":orange[**{esi} · ELEVATED**]"
    return f"**{esi}**"


def _render_queue(cards: list[PatientCard], key_prefix: str = "q") -> None:
    """Render a patient list with a View button per row to select a patient."""
    if not cards:
        st.info("No patients to display.")
        return

    # Three columns, not six: this table renders inside the narrower left pane,
    # and a six-way split clipped the View button down to a single glyph. The
    # button now shares the patient column instead of competing for width.
    widths = [4, 3, 2]

    h0, h1, h2 = st.columns(widths)
    h0.markdown("**Patient**")
    h1.markdown("**Chief Complaint**")
    h2.markdown("**Status**")

    for card in cards:
        p = card.patient
        st.divider()
        c1, c2, c3 = st.columns(widths)

        with c1:
            tag = ' <span class="returning-tag">RET</span>' if p.is_returning else ""
            st.markdown(f"**{p.name}**{tag}", unsafe_allow_html=True)
            st.caption(f"{p.patient_id} · {p.age} yrs")
            if st.button("Open", key=f"{key_prefix}_{p.patient_id}", use_container_width=True):
                st.session_state.selected_id = p.patient_id
                st.rerun()
        with c2:
            st.markdown(_truncate(p.chief_complaint))
        with c3:
            st.markdown(_queue_status(card))


# ─── Flash Messages ───────────────────────────────────────────────────────────

_FLASH_RENDERERS = {
    "success": st.sidebar.success,
    "warning": st.sidebar.warning,
    "error": st.sidebar.error,
}


def _set_flash(kind: str, message: str) -> None:
    """Queue a sidebar message to survive the next st.rerun()."""
    st.session_state.flash = (kind, message)


def _render_flash() -> None:
    """Render and consume any queued flash message."""
    flash = st.session_state.pop("flash", None)
    if flash is None:
        return
    kind, message = flash
    _FLASH_RENDERERS.get(kind, st.sidebar.info)(message)


# ─── New Patient Processing ───────────────────────────────────────────────────

def _process_new_patient(
    name: str,
    age: int,
    weight_kg: float,
    chief_complaint: str,
    hr: int,
    sbp: int,
    dbp: int,
    rr: int,
    spo2: float,
    temp: float,
) -> None:
    """Validate inputs, run the triage pipeline, persist the result."""
    try:
        vitals = VitalSigns(
            heart_rate=hr,
            systolic_bp=sbp,
            diastolic_bp=dbp,
            respiratory_rate=rr,
            spo2=spo2,
            temperature_c=temp,
        )
        patient = Patient(
            patient_id="PENDING",
            name=name,
            age=age,
            weight_kg=weight_kg,
            chief_complaint=chief_complaint,
            vitals=vitals,
        )

        patient = store.check_in(patient)

        with st.spinner(f"Triaging {patient.name} — {patient.patient_id}..."):
            card = run_triage(patient)

        store.save_card(card)
        st.session_state.selected_id = card.patient.patient_id

        # Flash messages go through session state: st.rerun() below discards
        # anything written directly to the sidebar, so the old direct calls
        # were never visible to the user.
        if card.has_system_error:
            _set_flash("error", f"{patient.patient_id}: automated triage failed. Triage manually.")
        elif card.needs_immediate_attention:
            _set_flash("error", f"{patient.patient_id} — {card.display_esi}, physician needed now.")
        elif card.is_escalated:
            _set_flash("warning", f"{patient.patient_id} — {card.display_esi}, elevated concern.")
        else:
            _set_flash("success", f"{patient.patient_id} triaged — {card.display_esi}")

        st.rerun()

    except Exception as e:
        _set_flash("error", f"Triage error: {e}")


# ─── Patient Search ───────────────────────────────────────────────────────────

def _run_search(query: str) -> None:
    """Search by patient ID (PT-XXXX) or partial name."""
    if query.upper().startswith("PT-"):
        card = store.search_by_id(query)
        if card:
            st.session_state.selected_id = card.patient.patient_id
            st.rerun()
            return

    results = store.search_by_name(query)
    if results:
        st.session_state.selected_id = results[0].patient.patient_id
        st.rerun()
    else:
        st.sidebar.warning(f"No patient found for '{query}'.")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Intake form and patient search in the sidebar."""
    st.sidebar.title("ED Triage System")
    st.sidebar.caption("Emergency Department — Patient Intake")
    _render_flash()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### New Patient Check-In")

    with st.sidebar.form("intake_form", clear_on_submit=True):
        name            = st.text_input("Full Name")
        col_a, col_w    = st.columns(2)
        with col_a:
            age         = st.number_input("Age (yrs)",    min_value=0,   max_value=130,   value=35)
        with col_w:
            weight_kg   = st.number_input("Weight (kg)",  min_value=0.5, max_value=500.0, value=70.0, step=0.5)
        chief_complaint = st.text_area(
            "Chief Complaint",
            placeholder="Patient's primary complaint in their own words...",
            height=80,
        )

        st.markdown("**Vital Signs**")
        col1, col2 = st.columns(2)
        with col1:
            hr  = st.number_input("Heart Rate (bpm)",    min_value=0,   max_value=300,   value=80)
            sbp = st.number_input("Systolic BP (mmHg)",  min_value=0,   max_value=300,   value=120)
            dbp = st.number_input("Diastolic BP (mmHg)", min_value=0,   max_value=200,   value=80)
        with col2:
            rr   = st.number_input("Resp Rate (brpm)",   min_value=0,   max_value=60,    value=16)
            spo2 = st.number_input("SpO2 (%)",           min_value=0.0, max_value=100.0, value=98.0, step=0.5)
            temp = st.number_input("Temp (°C)",          min_value=25.0,max_value=45.0,  value=37.0, step=0.1)

        submitted = st.form_submit_button(
            "Check In & Triage", use_container_width=True, type="primary"
        )

    if submitted:
        if not name.strip():
            st.sidebar.error("Patient name is required.")
        elif not chief_complaint.strip():
            st.sidebar.error("Chief complaint is required.")
        else:
            _process_new_patient(
                name.strip(), age, weight_kg,
                chief_complaint.strip(),
                hr, sbp, dbp, rr, spo2, temp,
            )

    # ── Search ────────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Find Patient")
    query = st.sidebar.text_input("ID or Name", placeholder="PT-0001 or last name...")
    if st.sidebar.button("Search", use_container_width=True):
        if query.strip():
            _run_search(query.strip())
        else:
            st.sidebar.warning("Enter a patient ID or name to search.")


# ─── Main Layout ──────────────────────────────────────────────────────────────

def main() -> None:
    grounded = _init_rag()

    _render_sidebar()

    # Kept in the sidebar: as a main-pane banner this re-rendered on every
    # interaction and pushed the dashboard down the page each time.
    if not grounded:
        st.sidebar.markdown("---")
        st.sidebar.info(
            "No clinical knowledge base loaded. Triage still runs, but reasoning "
            "is not grounded in retrieved guidelines. Add PDFs to `/data` and run "
            "`python -m rag.ingest` to enable grounding."
        )

    # ── Dashboard Header ──────────────────────────────────────────────────────
    st.title("Emergency Department Triage")

    stats = store.queue_stats()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("In Department", stats["active"])
    s2.metric("Physician Now", stats["immediate"])
    s3.metric("Elevated",      stats["elevated"])
    s4.metric("Routine",       stats["routine"])

    # ── Active Alerts ─────────────────────────────────────────────────────────
    #
    # Scoped to the active queue and to IMMEDIATE only. Banners previously drew
    # from every card ever created, so they accumulated for the life of the
    # store and pushed the dashboard off screen.
    active = store.get_queue()
    errored = [c for c in active if c.has_system_error]
    # System errors escalate to IMMEDIATE too, but they are a pipeline failure,
    # not a clinical judgment, so they get their own notice rather than a
    # physician alert that implies the system assessed the patient.
    immediate = [
        c for c in active
        if c.needs_immediate_attention and not c.has_system_error
    ]

    for c in immediate:
        st.error(
            f"PHYSICIAN ALERT — {c.patient.name}  |  {c.patient.patient_id}  |  "
            f"{c.display_esi}  |  {_truncate(c.patient.chief_complaint, 80)}"
        )
    if errored:
        st.warning(
            f"{len(errored)} patient(s) could not be triaged automatically and need "
            f"manual triage: {', '.join(c.patient.patient_id for c in errored)}"
        )

    st.divider()

    # ── Two-Column Layout: Queue | Patient Card ────────────────────────────────
    left, right = st.columns([2, 3])

    with left:
        tab_active, tab_all = st.tabs(["Active Queue", "All Patients"])

        with tab_active:
            # Sort by urgency first, then by ESI, then by wait time. Patients
            # needing a physician now sort above everyone else.
            _ESCALATION_RANK = {
                EscalationLevel.IMMEDIATE: 0,
                EscalationLevel.ELEVATED: 1,
                EscalationLevel.NONE: 2,
            }
            queue = sorted(
                active,
                key=lambda c: (
                    _ESCALATION_RANK[c.escalation.level],
                    c.esi_score if c.esi_score is not None else 0,
                    c.patient.check_in_time,
                ),
            )
            _render_queue(queue, key_prefix="active")

        with tab_all:
            _render_queue(store.get_all_cards(), key_prefix="all")

    with right:
        selected_id = st.session_state.get("selected_id")
        if selected_id:
            card = store.get_card(selected_id)
            if card:
                _render_patient_card(card)
            else:
                st.warning(f"Patient {selected_id} not found in store.")
        else:
            st.info(
                "Check in a new patient or select one from the queue to view their triage card."
            )

    # ── End-of-Shift Admin ────────────────────────────────────────────────────
    st.divider()
    with st.expander("Admin — End of Shift Reset"):
        st.warning("This action permanently deletes all patient records and cannot be undone.")
        confirm = st.checkbox("I confirm I want to clear all patient records.")
        if st.button("Clear All Records", disabled=not confirm, type="secondary"):
            store.clear_all()
            st.session_state.pop("selected_id", None)
            st.success("All records cleared. Ready for new shift.")
            st.rerun()


if __name__ == "__main__":
    main()
