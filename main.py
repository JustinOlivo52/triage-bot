"""
main.py — Streamlit UI for the ED Triage System.

V2 Phase 4: this is now a thin client of backend/'s API (api_client.py) —
it no longer imports agents/ or memory/ at all. Every action, including
reading the queue, is a real HTTP call carrying a JWT, the same as any other
API consumer would make.
"""

import logging

import streamlit as st

import api_client
from config import DEMO_MODE, LIVE_TRIAGE_LIMIT
from models import (
    EscalationLevel,
    PatientCard,
    SymptomStatus,
    TriageStatus,
)

logger = logging.getLogger(__name__)

# The demo accounts' password — must match DEMO_ACCOUNT_PASSWORD's default in
# backend/core/config.py. Shown on the login screen only, never fetched from
# the API (an endpoint that reveals passwords would be its own problem).
_DEMO_PASSWORD_HINT = "demo1234"

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

_SYMPTOM_STATUS_MARK = {
    SymptomStatus.PRESENT: "🔴",
    SymptomStatus.DENIED: "⚪",
    SymptomStatus.HISTORICAL: "🕓",
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


# ─── Session / Auth ────────────────────────────────────────────────────────────

def _current_session() -> api_client.Session | None:
    return st.session_state.get("session")


def _login_screen() -> None:
    st.title("ED Triage System")
    st.caption("Emergency Department — sign in to continue")

    if DEMO_MODE:
        st.info(
            "**Demo credentials** — feel free to explore:\n\n"
            f"- Nurse: `demo_nurse` / `{_DEMO_PASSWORD_HINT}`\n"
            f"- Physician: `demo_physician` / `{_DEMO_PASSWORD_HINT}`\n"
            f"- Admin: `demo_admin` / `{_DEMO_PASSWORD_HINT}`\n\n"
            "One shared department queue — the same as real staff would see."
        )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

    if submitted:
        if not username.strip() or not password:
            st.error("Enter a username and password.")
        else:
            try:
                st.session_state["session"] = api_client.login(username.strip(), password)
                st.rerun()
            except api_client.ApiError as e:
                st.error(f"Login failed: {e}")


def _logout() -> None:
    st.session_state.clear()
    st.rerun()


# ─── Demo Guardrails ──────────────────────────────────────────────────────────
#
# Live-triage runs still cost real API money even for a logged-in user, so
# the per-browser-session cap from V1 survives the move to real auth — it's
# a UI-side guardrail on top of, not instead of, login.

def _live_runs_used() -> int:
    return st.session_state.get("_live_runs", 0)


def _live_runs_remaining() -> int | None:
    """Live triage runs left this session, or None when uncapped."""
    if not DEMO_MODE:
        return None
    return max(0, LIVE_TRIAGE_LIMIT - _live_runs_used())


def _record_live_run() -> None:
    st.session_state["_live_runs"] = _live_runs_used() + 1


# ─── Vitals Renderer ─────────────────────────────────────────────────────────

def _render_vitals(card: PatientCard) -> None:
    """
    Render vitals in two columns, with any abnormal findings called out below.

    V1 recomputed per-field severity client-side (agents.assessment.
    vital_severity_map) to color each value individually. That function is
    clinical assessment logic, and Phase 4's whole point is that the UI no
    longer imports agents/ at all — duplicating threshold logic here to
    re-derive per-field color would be exactly the kind of drift CLAUDE.md's
    "single source of truth for clinical thresholds" rule exists to prevent.
    The backend's own findings (already computed, already on the card) are
    shown as a callout instead of being re-derived.
    """
    v = card.patient.vitals
    vital_findings = [f.detail for f in card.escalation.findings if f.category.value == "vital"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Heart Rate:** {v.heart_rate} bpm")
        st.markdown(f"**Blood Pressure:** {v.bp_display}")
        st.markdown(f"**Resp Rate:** {v.respiratory_rate} breaths/min")
    with col2:
        st.markdown(f"**SpO2:** {v.spo2}%")
        st.markdown(f"**Temperature:** {v.temperature_c}°C / {v.temperature_f}°F")

    if vital_findings:
        st.markdown(f'<span class="vital-critical">⚠ {"; ".join(vital_findings)}</span>', unsafe_allow_html=True)


# ─── Patient Card Renderer ────────────────────────────────────────────────────

def _render_patient_card(session: api_client.Session, view: api_client.EncounterView) -> None:
    """Render the full structured patient card for the healthcare team."""
    card = view.card
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
        if p.status is TriageStatus.RESOLVED:
            st.caption("✅ Resolved")

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

        # Shows what the system believed it read in the complaint, and which of
        # those actually carried clinical weight. A denied or historical symptom
        # is recorded but deliberately produces no finding.
        if result.extracted_symptoms:
            with st.expander(f"Symptom Extraction ({len(result.extracted_symptoms)})"):
                st.caption(
                    "Only symptoms marked *present* become findings. Denied and "
                    "historical mentions are recorded but carry no clinical weight."
                )
                for extracted in result.extracted_symptoms:
                    st.markdown(
                        f"{_SYMPTOM_STATUS_MARK[extracted.status]} "
                        f"**{extracted.symptom}** — {extracted.status.value}"
                    )

    # ── Prior Visit History ────────────────────────────────────────────────────
    if p.is_returning:
        try:
            prior = api_client.prior_visits(session, view.encounter_id)
        except api_client.ApiError as e:
            prior = []
            st.caption(f"Could not load prior visits: {e}")
        if prior:
            with st.expander(f"Prior Visits ({len(prior)})"):
                for pv in prior:
                    st.markdown(
                        f"**{pv.card.patient.check_in_time.strftime('%b %d, %Y  %H:%M')}**  —  "
                        f"{pv.card.patient.chief_complaint}  —  {pv.card.display_esi}"
                    )

    # ── Disposition ─────────────────────────────────────────────────────────────
    #
    # Resolving is physician/admin, enforced server-side (backend/api/deps.py
    # require_role); the form is only shown to roles that can actually submit
    # it. This is the nurse-override/disposition workflow CLAUDE.md listed as
    # a known gap — folded into Phase 4 since there was no route to audit
    # until it existed.
    if p.status is not TriageStatus.RESOLVED and session.role in ("physician", "admin"):
        st.divider()
        with st.form(f"resolve_form_{view.encounter_id}"):
            note = st.text_input("Disposition note (optional)", placeholder="e.g. Discharged home, follow up PCP")
            if st.form_submit_button("Mark Resolved"):
                try:
                    api_client.resolve(session, view.encounter_id, note.strip() or None)
                    st.success(f"{p.patient_id} marked resolved.")
                    st.rerun()
                except api_client.ApiError as e:
                    st.error(f"Could not resolve: {e}")


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


def _render_queue(views: list[api_client.EncounterView], key_prefix: str = "q") -> None:
    """Render a patient list with a View button per row to select a patient."""
    if not views:
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

    for view in views:
        card = view.card
        p = card.patient
        st.divider()
        c1, c2, c3 = st.columns(widths)

        with c1:
            tag = ' <span class="returning-tag">RET</span>' if p.is_returning else ""
            st.markdown(f"**{p.name}**{tag}", unsafe_allow_html=True)
            st.caption(f"{p.patient_id} · {p.age} yrs")
            if st.button("Open", key=f"{key_prefix}_{view.encounter_id}", use_container_width=True):
                st.session_state.selected_id = view.encounter_id
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
    session: api_client.Session,
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
    """Send intake data to the backend and let it run the full pipeline."""
    # The cap is checked before the request goes out, so a refused run never
    # even reaches the backend.
    remaining = _live_runs_remaining()
    if remaining is not None and remaining <= 0:
        _set_flash(
            "warning",
            f"Demo limit reached — {LIVE_TRIAGE_LIMIT} live triage runs per session. "
            f"The seeded patients are still fully explorable.",
        )
        # Rerun so the message is actually seen. _render_flash() already ran
        # earlier in this pass, so without this the refusal is silent and the
        # button looks broken.
        st.rerun()

    try:
        with st.spinner(f"Triaging {name}..."):
            view = api_client.check_in(
                session,
                full_name=name, age=age, weight_kg=weight_kg,
                chief_complaint=chief_complaint,
                vitals={
                    "heart_rate": hr, "systolic_bp": sbp, "diastolic_bp": dbp,
                    "respiratory_rate": rr, "spo2": spo2, "temperature_c": temp,
                },
            )
        _record_live_run()

        card = view.card
        st.session_state.selected_id = view.encounter_id

        # Flash messages go through session state: st.rerun() below discards
        # anything written directly to the sidebar, so a direct call here
        # would never be visible to the user.
        if card.has_system_error:
            _set_flash("error", f"{card.patient.patient_id}: automated triage failed. Triage manually.")
        elif card.needs_immediate_attention:
            _set_flash("error", f"{card.patient.patient_id} — {card.display_esi}, physician needed now.")
        elif card.is_escalated:
            _set_flash("warning", f"{card.patient.patient_id} — {card.display_esi}, elevated concern.")
        else:
            _set_flash("success", f"{card.patient.patient_id} triaged — {card.display_esi}")

        st.rerun()

    except api_client.ApiError as e:
        _set_flash("error", f"Triage error: {e}")


# ─── Patient Search ───────────────────────────────────────────────────────────

def _run_search(session: api_client.Session, query: str) -> None:
    """Search by patient ID (PT-XXXX) or partial name."""
    try:
        if query.upper().startswith("PT-"):
            view = api_client.get_by_patient_identifier(session, query)
            if view:
                st.session_state.selected_id = view.encounter_id
                st.rerun()
                return

        results = api_client.search(session, query)
        if results:
            st.session_state.selected_id = results[0].encounter_id
            st.rerun()
        else:
            st.sidebar.warning(f"No patient found for '{query}'.")
    except api_client.ApiError as e:
        st.sidebar.error(f"Search failed: {e}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _render_sidebar(session: api_client.Session) -> None:
    """Account info, intake form, and patient search in the sidebar."""
    st.sidebar.title("ED Triage System")
    st.sidebar.caption(f"Signed in as **{session.username}** ({session.role})")
    if st.sidebar.button("Log Out", use_container_width=True):
        _logout()

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
        remaining = _live_runs_remaining()
        if remaining is not None:
            st.caption(
                f"Demo: {remaining} of {LIVE_TRIAGE_LIMIT} live triage runs left "
                f"this session."
            )

    if submitted:
        if not name.strip():
            st.sidebar.error("Patient name is required.")
        elif not chief_complaint.strip():
            st.sidebar.error("Chief complaint is required.")
        else:
            _process_new_patient(
                session,
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
            _run_search(session, query.strip())
        else:
            st.sidebar.warning("Enter a patient ID or name to search.")


# ─── Main Layout ──────────────────────────────────────────────────────────────

def _compute_stats(all_views: list[api_client.EncounterView], active_views: list[api_client.EncounterView]) -> dict:
    """Same summary the old PatientStore.queue_stats() produced, computed
    client-side over one fetched list instead of a server-side aggregate —
    the queue is small enough (a department's worth of patients) that this
    costs nothing and avoids a dedicated stats endpoint for one dashboard."""
    return {
        "total": len(all_views),
        "active": len(active_views),
        "immediate": sum(1 for v in active_views if v.card.needs_immediate_attention),
        "elevated": sum(
            1 for v in active_views
            if v.card.is_escalated and not v.card.needs_immediate_attention
        ),
        "routine": sum(1 for v in active_views if not v.card.is_escalated and v.card.triage_result),
        "pending": sum(1 for v in active_views if v.card.patient.status is TriageStatus.PENDING),
        "system_errors": sum(1 for v in all_views if v.card.has_system_error),
    }


def main() -> None:
    session = _current_session()
    if session is None:
        _login_screen()
        return

    mode = api_client.retrieval_mode()

    _render_sidebar(session)

    # Kept in the sidebar: as a main-pane banner this re-rendered on every
    # interaction and pushed the dashboard down the page each time.
    if mode != "semantic":
        st.sidebar.markdown("---")
        if mode == "lexical":
            st.sidebar.info(
                "Retrieval is running in **lexical** mode. Reasoning is grounded "
                "in the clinical reference, but matching is keyword-based. Set "
                "`VOYAGE_API_KEY` to enable semantic search."
            )
        else:
            st.sidebar.info(
                "No clinical reference index loaded. Triage still runs, but "
                "reasoning is not grounded in retrieved criteria."
            )

    # ── Dashboard Header ──────────────────────────────────────────────────────
    st.title("Emergency Department Triage")

    try:
        all_views = api_client.list_encounters(session)
    except api_client.ApiError as e:
        st.error(f"Could not reach the backend: {e}")
        return

    active = [v for v in all_views if v.card.patient.status is not TriageStatus.RESOLVED]
    stats = _compute_stats(all_views, active)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("In Department", stats["active"])
    s2.metric("Physician Now", stats["immediate"])
    s3.metric("Elevated",      stats["elevated"])
    s4.metric("Routine",       stats["routine"])

    # ── Active Alerts ─────────────────────────────────────────────────────────
    #
    # Scoped to the active queue and to IMMEDIATE only, so banners don't
    # accumulate for the life of the department and push the dashboard off
    # screen.
    errored = [v for v in active if v.card.has_system_error]
    # System errors escalate to IMMEDIATE too, but they are a pipeline failure,
    # not a clinical judgment, so they get their own notice rather than a
    # physician alert that implies the system assessed the patient.
    immediate = [
        v for v in active
        if v.card.needs_immediate_attention and not v.card.has_system_error
    ]

    for v in immediate:
        p = v.card.patient
        st.error(
            f"PHYSICIAN ALERT — {p.name}  |  {p.patient_id}  |  "
            f"{v.card.display_esi}  |  {_truncate(p.chief_complaint, 80)}"
        )
    if errored:
        st.warning(
            f"{len(errored)} patient(s) could not be triaged automatically and need "
            f"manual triage: {', '.join(v.card.patient.patient_id for v in errored)}"
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
                key=lambda v: (
                    _ESCALATION_RANK[v.card.escalation.level],
                    v.card.esi_score if v.card.esi_score is not None else 0,
                    v.card.patient.check_in_time,
                ),
            )
            _render_queue(queue, key_prefix="active")

        with tab_all:
            _render_queue(all_views, key_prefix="all")

    with right:
        selected_id = st.session_state.get("selected_id")
        selected_view = next((v for v in all_views if v.encounter_id == selected_id), None) if selected_id else None
        if selected_view:
            _render_patient_card(session, selected_view)
        elif selected_id:
            st.warning("That patient is not in the current queue.")
        else:
            st.info(
                "Check in a new patient or select one from the queue to view their triage card."
            )

    # ── End-of-Shift Admin ────────────────────────────────────────────────────
    if session.role == "admin":
        st.divider()
        with st.expander("Admin — End of Shift Reset"):
            st.warning(
                "This permanently deletes all patient records and cannot be undone. "
                "The audit log is never deleted by this — including the record of this reset."
            )
            confirm = st.checkbox("I confirm I want to clear all patient records.")
            if st.button("Clear All Records", disabled=not confirm, type="secondary"):
                try:
                    api_client.reset_queue(session)
                    st.session_state.pop("selected_id", None)
                    st.success("Reset complete.")
                    st.rerun()
                except api_client.ApiError as e:
                    st.error(f"Reset failed: {e}")


if __name__ == "__main__":
    main()
