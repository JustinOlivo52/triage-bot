"""
main.py — Streamlit UI for the ED Triage System.
"""

import streamlit as st

from config import VITAL_THRESHOLDS
from rag.ingest import ingest, vector_store_exists
from agents.triage_agent import run_triage
from memory.patient_store import store
from models import Patient, VitalSigns, PatientCard, TriageStatus

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="ED Triage System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.red-flag-header {
    background-color: #c0392b; color: white;
    padding: 10px 16px; border-radius: 6px;
    font-size: 1.15rem; font-weight: bold; margin-bottom: 10px;
}
.esi-badge {
    display: inline-block;
    padding: 5px 14px; border-radius: 5px;
    font-weight: bold; font-size: 1rem; color: white;
}
.vital-ok  { color: #27ae60; font-weight: 500; }
.vital-bad { color: #c0392b; font-weight: bold; }
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


def _esi_badge(score: int) -> str:
    """Return an HTML badge string for the given ESI score."""
    color = _ESI_COLORS.get(score, "#7f8c8d")
    label = _ESI_LABELS.get(score, f"ESI {score}")
    return f'<span class="esi-badge" style="background:{color};">{label}</span>'


# ─── RAG Initialization ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading clinical knowledge base...")
def _init_rag():
    """Build or load the ChromaDB vector store once per session."""
    return ingest()


# ─── Vitals Renderer ─────────────────────────────────────────────────────────

def _render_vitals(card: PatientCard) -> None:
    """Render vitals in two columns, highlighting out-of-range values in red."""
    v = card.patient.vitals

    abnormal: set[str] = set()
    if v.heart_rate > VITAL_THRESHOLDS["hr_high"] or v.heart_rate < VITAL_THRESHOLDS["hr_low"]:
        abnormal.add("hr")
    if v.respiratory_rate > VITAL_THRESHOLDS["rr_high"]:
        abnormal.add("rr")
    if v.spo2 < VITAL_THRESHOLDS["spo2_low"]:
        abnormal.add("spo2")
    if v.temperature_c > VITAL_THRESHOLDS["temp_high_c"] or v.temperature_c < VITAL_THRESHOLDS["temp_low_c"]:
        abnormal.add("temp")
    if v.systolic_bp > VITAL_THRESHOLDS["sbp_high"] or v.systolic_bp < VITAL_THRESHOLDS["sbp_low"]:
        abnormal.add("bp")

    def _span(key: str, text: str) -> str:
        css = "vital-bad" if key in abnormal else "vital-ok"
        return f'<span class="{css}">{text}</span>'

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Heart Rate:** {_span('hr', f'{v.heart_rate} bpm')}", unsafe_allow_html=True)
        st.markdown(f"**Blood Pressure:** {_span('bp', v.bp_display)}", unsafe_allow_html=True)
        st.markdown(f"**Resp Rate:** {_span('rr', f'{v.respiratory_rate} breaths/min')}", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**SpO2:** {_span('spo2', f'{v.spo2}%')}", unsafe_allow_html=True)
        st.markdown(f"**Temperature:** {_span('temp', f'{v.temperature_c}°C / {v.temperature_f}°F')}", unsafe_allow_html=True)


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
        if card.is_red_flagged:
            st.markdown('<div class="red-flag-header">RED FLAG</div>', unsafe_allow_html=True)
        elif card.triage_result:
            st.markdown(_esi_badge(card.triage_result.esi_score), unsafe_allow_html=True)
        else:
            st.info("Pending Triage")

    st.divider()

    # ── Chief Complaint ────────────────────────────────────────────────────────
    st.markdown(f"**Chief Complaint:** {p.chief_complaint.title()}")
    st.markdown("**Vital Signs**")
    _render_vitals(card)

    st.divider()

    # ── Red Flag Physician Summary ─────────────────────────────────────────────
    if card.is_red_flagged:
        alert   = card.red_flag_alert
        summary = alert.physician_summary

        st.error(f"PHYSICIAN ALERT  —  {alert.layer.value if alert.layer else 'Red Flag Triggered'}")

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

    # ── Standard Triage Result ─────────────────────────────────────────────────
    elif card.triage_result:
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
                        f"{visit.patient.chief_complaint.title()}  —  {visit.display_esi}"
                    )


# ─── Queue Renderer ───────────────────────────────────────────────────────────

def _render_queue(cards: list[PatientCard], key_prefix: str = "q") -> None:
    """Render a patient list with a View button per row to select a patient."""
    if not cards:
        st.info("No patients to display.")
        return

    # Column headers
    h0, h1, h2, h3, h4, h5 = st.columns([0.6, 1.1, 2, 0.8, 3, 1.5])
    h1.markdown("**ID**")
    h2.markdown("**Name**")
    h3.markdown("**Age**")
    h4.markdown("**Chief Complaint**")
    h5.markdown("**Status**")
    st.divider()

    for card in cards:
        p = card.patient

        if card.is_red_flagged:
            status_label = "RED FLAG"
        elif card.patient.status == TriageStatus.TRIAGED and card.triage_result:
            status_label = f"ESI {card.triage_result.esi_score}"
        else:
            status_label = "Pending"

        c0, c1, c2, c3, c4, c5 = st.columns([0.6, 1.1, 2, 0.8, 3, 1.5])

        with c0:
            if st.button("View", key=f"{key_prefix}_{p.patient_id}"):
                st.session_state.selected_id = p.patient_id
                st.rerun()
        with c1:
            st.markdown(f"`{p.patient_id}`")
        with c2:
            tag = " *" if p.is_returning else ""
            st.markdown(f"{p.name}{tag}")
        with c3:
            st.markdown(str(p.age))
        with c4:
            st.markdown(p.chief_complaint.title()[:55])
        with c5:
            if card.is_red_flagged:
                st.markdown(f"**:red[{status_label}]**")
            elif status_label.startswith("ESI"):
                st.markdown(f"**{status_label}**")
            else:
                st.markdown(status_label)


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

        if card.is_red_flagged:
            st.sidebar.error(
                f"RED FLAG — {patient.patient_id} flagged for immediate physician attention."
            )
        else:
            st.sidebar.success(f"{patient.patient_id} triaged — {card.display_esi}")

        st.rerun()

    except Exception as e:
        st.sidebar.error(f"Triage error: {e}")


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
    _init_rag()

    if not vector_store_exists():
        st.warning(
            "Clinical knowledge base not found. Place ESI guideline PDFs in /data and "
            "run `python -m rag.ingest` to build it. The system will still triage "
            "patients but LLM reasoning will not be grounded in clinical documents."
        )

    _render_sidebar()

    # ── Dashboard Header ──────────────────────────────────────────────────────
    st.title("Emergency Department Triage")

    stats = store.queue_stats()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Patients",  stats["total"])
    s2.metric("Red Flagged",     stats["red_flagged"])
    s3.metric("Triaged",         stats["triaged"])
    s4.metric("Pending",         stats["pending"])

    # ── Active Red Flag Alerts ────────────────────────────────────────────────
    flagged = [c for c in store.get_all_cards() if c.is_red_flagged]
    if flagged:
        for c in flagged:
            st.error(
                f"PHYSICIAN ALERT — {c.patient.name}  |  {c.patient.patient_id}  |  "
                f"Chief Complaint: {c.patient.chief_complaint.title()}  |  "
                f"Trigger: {c.red_flag_alert.layer.value if c.red_flag_alert.layer else 'Red Flag Triggered'}"
            )

    st.divider()

    # ── Two-Column Layout: Queue | Patient Card ────────────────────────────────
    left, right = st.columns([2, 3])

    with left:
        tab_active, tab_all = st.tabs(["Active Queue", "All Patients"])

        with tab_active:
            # Red flagged patients always sort to the top
            queue = sorted(
                store.get_queue(),
                key=lambda c: (not c.is_red_flagged, c.patient.check_in_time),
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
