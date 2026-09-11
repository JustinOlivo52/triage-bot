"""
agents/assessment.py — Deterministic clinical assessment and escalation rules.

Pure logic only: no LLM calls, no network, no API key. Every function here is a
deterministic function of its inputs, which keeps the clinical rules cheap to
run and testable without the model stack installed.

    assess_patient()    → list[ClinicalFinding]   from vitals, symptoms, age
    derive_escalation() → EscalationLevel         from ESI score + findings

Findings carry a severity so "HR 104" and "HR 165" are not treated as the same
event. A CRITICAL finding escalates on its own; CONCERNING findings raise the
patient's visibility without claiming they cannot wait.
"""

import logging
from typing import NamedTuple, Optional

from config import (
    AGE_THRESHOLDS,
    CONCERNING_SYMPTOMS,
    CONCERNING_VITALS,
    CRITICAL_SYMPTOMS,
    CRITICAL_VITALS,
    ESI_IMMEDIATE_MAX,
)
from models import (
    ClinicalFinding,
    EscalationLevel,
    FindingCategory,
    FindingSeverity,
    Patient,
    VitalSigns,
)

logger = logging.getLogger(__name__)


# ─── Vital Sign Rules ─────────────────────────────────────────────────────────


class _VitalRule(NamedTuple):
    """
    One vital sign and the labels used to describe it out of range.

    `key` is the prefix used in CRITICAL_VITALS / CONCERNING_VITALS, so a rule
    with key "hr" reads `hr_high` and `hr_low` from both tiers. A label of None
    means that direction is not evaluated (SpO2 has no dangerous upper bound).
    """

    attr: str            # attribute name on VitalSigns
    key: str             # threshold key prefix
    display: str         # short label, e.g. "HR"
    unit: str
    high_label: Optional[str]
    low_label: Optional[str]
    critical_high_label: Optional[str]
    critical_low_label: Optional[str]


_VITAL_RULES: tuple[_VitalRule, ...] = (
    _VitalRule("heart_rate", "hr", "HR", "bpm",
               "Tachycardia", "Bradycardia",
               "Marked tachycardia", "Marked bradycardia"),
    _VitalRule("respiratory_rate", "rr", "RR", "breaths/min",
               "Tachypnea", "Low respiratory rate",
               "Severe tachypnea", "Respiratory depression"),
    _VitalRule("spo2", "spo2", "SpO2", "%",
               None, "Borderline oxygenation",
               None, "Significant hypoxia"),
    _VitalRule("temperature_c", "temp", "Temp", "°C",
               "Fever", "Low temperature",
               "Hyperpyrexia", "Hypothermia"),
    _VitalRule("systolic_bp", "sbp", "SBP", "mmHg",
               "Hypertensive urgency", "Borderline hypotension",
               "Severe hypertension", "Hypotension"),
    _VitalRule("diastolic_bp", "dbp", "DBP", "mmHg",
               "Diastolic hypertension", None,
               "Severe diastolic hypertension", None),
)


def _vital_finding(
    rule: _VitalRule,
    value: float,
    severity: FindingSeverity,
    label: str,
    bound: str,
) -> ClinicalFinding:
    return ClinicalFinding(
        severity=severity,
        category=FindingCategory.VITAL,
        detail=f"{label} — {rule.display} {value} {rule.unit} (threshold {bound})",
    )


def _evaluate_vital(rule: _VitalRule, value: float) -> Optional[ClinicalFinding]:
    """
    Evaluate one vital against both tiers, most severe first.

    Returns at most one finding — a heart rate of 165 is marked tachycardia, not
    marked tachycardia *and* tachycardia.
    """
    critical_high = CRITICAL_VITALS.get(f"{rule.key}_high")
    critical_low = CRITICAL_VITALS.get(f"{rule.key}_low")
    concerning_high = CONCERNING_VITALS.get(f"{rule.key}_high")
    concerning_low = CONCERNING_VITALS.get(f"{rule.key}_low")

    if rule.critical_high_label and critical_high is not None and value > critical_high:
        return _vital_finding(rule, value, FindingSeverity.CRITICAL,
                              rule.critical_high_label, f">{critical_high}")
    if rule.critical_low_label and critical_low is not None and value < critical_low:
        return _vital_finding(rule, value, FindingSeverity.CRITICAL,
                              rule.critical_low_label, f"<{critical_low}")
    if rule.high_label and concerning_high is not None and value > concerning_high:
        return _vital_finding(rule, value, FindingSeverity.CONCERNING,
                              rule.high_label, f">{concerning_high}")
    if rule.low_label and concerning_low is not None and value < concerning_low:
        return _vital_finding(rule, value, FindingSeverity.CONCERNING,
                              rule.low_label, f"<{concerning_low}")
    return None


def assess_vitals(vitals: VitalSigns) -> list[ClinicalFinding]:
    """Compare every vital sign against the critical and concerning tiers."""
    findings: list[ClinicalFinding] = []
    for rule in _VITAL_RULES:
        finding = _evaluate_vital(rule, getattr(vitals, rule.attr))
        if finding is not None:
            findings.append(finding)
    return findings


def vital_severity_map(vitals: VitalSigns) -> dict[str, FindingSeverity]:
    """
    Map each out-of-range vital attribute to its severity.

    Used by the UI so display highlighting reads the same thresholds as the
    clinical logic instead of keeping its own copy.
    """
    severities: dict[str, FindingSeverity] = {}
    for rule in _VITAL_RULES:
        finding = _evaluate_vital(rule, getattr(vitals, rule.attr))
        if finding is not None:
            severities[rule.attr] = finding.severity
    return severities


def format_thresholds_for_prompt() -> str:
    """
    Render the vital sign thresholds as prompt text.

    Single source of truth: the LLM is told the same numbers the deterministic
    checks use, so the prompt cannot drift away from config.py.
    """
    def _bounds(table: dict, rule: _VitalRule) -> str:
        high = table.get(f"{rule.key}_high")
        low = table.get(f"{rule.key}_low")
        parts = []
        if high is not None and (rule.high_label or rule.critical_high_label):
            parts.append(f">{high}")
        if low is not None and (rule.low_label or rule.critical_low_label):
            parts.append(f"<{low}")
        return " or ".join(parts)

    lines: list[str] = []
    for rule in _VITAL_RULES:
        concerning = _bounds(CONCERNING_VITALS, rule)
        critical = _bounds(CRITICAL_VITALS, rule)
        lines.append(
            f"- {rule.display} ({rule.unit}): concerning {concerning}; critical {critical}"
        )
    return "\n".join(lines)


# ─── Symptom Rules ────────────────────────────────────────────────────────────


def assess_symptoms(chief_complaint: str) -> list[ClinicalFinding]:
    """
    Scan the chief complaint for critical and concerning symptom keywords.

    Known limitation: this is a substring scan, so it cannot tell "chest pain"
    from "denies chest pain". The keyword lists are kept deliberately narrow to
    limit the damage; negation and history are handled by structured extraction
    in the triage node rather than here.
    """
    complaint = chief_complaint.lower()
    findings: list[ClinicalFinding] = []

    for symptom in CRITICAL_SYMPTOMS:
        if symptom in complaint:
            findings.append(ClinicalFinding(
                severity=FindingSeverity.CRITICAL,
                category=FindingCategory.SYMPTOM,
                detail=f"Critical symptom reported: '{symptom}'",
            ))

    for symptom in CONCERNING_SYMPTOMS:
        if symptom in complaint:
            findings.append(ClinicalFinding(
                severity=FindingSeverity.CONCERNING,
                category=FindingCategory.SYMPTOM,
                detail=f"Concerning symptom reported: '{symptom}'",
            ))

    return findings


# ─── Age Risk ─────────────────────────────────────────────────────────────────


def assess_age_risk(patient: Patient, other_findings: list[ClinicalFinding]) -> list[ClinicalFinding]:
    """
    Flag high-risk age groups that already have another abnormal finding.

    Age alone is not a finding. Age combined with something else is, because
    both ends of the age range blunt the usual warning signs.
    """
    if not other_findings:
        return []
    if patient.age_group not in ("pediatric", "geriatric"):
        return []

    reason = (
        "atypical presentation and adult vital ranges do not apply"
        if patient.age_group == "pediatric"
        else "blunted physiologic response can mask severity"
    )
    return [ClinicalFinding(
        severity=FindingSeverity.CONCERNING,
        category=FindingCategory.AGE,
        detail=(
            f"High-risk age group: {patient.age_group} (age {patient.age}) "
            f"with a concurrent abnormal finding — {reason}"
        ),
    )]


# ─── Full Deterministic Assessment ────────────────────────────────────────────


def assess_patient(patient: Patient) -> list[ClinicalFinding]:
    """
    Run every deterministic check. No LLM call, no API cost.

    Runs before triage reasoning so the findings can be handed to the LLM as
    context rather than competing with it.
    """
    findings = assess_vitals(patient.vitals)
    findings.extend(assess_symptoms(patient.chief_complaint))
    findings.extend(assess_age_risk(patient, findings))

    logger.info(
        "Assessment for %s — %d finding(s), %d critical",
        patient.patient_id,
        len(findings),
        sum(1 for f in findings if f.severity is FindingSeverity.CRITICAL),
    )
    return findings


# ─── Escalation Derivation ────────────────────────────────────────────────────


def derive_escalation(
    esi_score: Optional[int],
    findings: list[ClinicalFinding],
) -> tuple[EscalationLevel, list[str]]:
    """
    Decide how urgently this patient needs attention.

    Pure function — no I/O, no LLM, fully testable. The ESI score and the
    deterministic findings are both inputs; neither one suppresses the other.

    Rules, in order:
        1. ESI at or below ESI_IMMEDIATE_MAX  → IMMEDIATE
        2. Any CRITICAL finding               → IMMEDIATE
        3. Any remaining finding              → ELEVATED
        4. Otherwise                          → NONE
    """
    reasons: list[str] = []
    level = EscalationLevel.NONE

    if esi_score is not None and esi_score <= ESI_IMMEDIATE_MAX:
        level = EscalationLevel.IMMEDIATE
        reasons.append(
            f"ESI {esi_score} meets the immediate-attention threshold (ESI ≤ {ESI_IMMEDIATE_MAX})"
        )

    critical = [f for f in findings if f.severity is FindingSeverity.CRITICAL]
    if critical:
        level = EscalationLevel.IMMEDIATE
        reasons.extend(f.detail for f in critical)

    if level is EscalationLevel.NONE and findings:
        level = EscalationLevel.ELEVATED
        reasons.extend(f.detail for f in findings)

    return level, reasons
