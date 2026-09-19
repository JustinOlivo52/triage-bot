"""
Tests for the deterministic clinical rules in agents/assessment.py.

These deliberately require no API key, no network, and no embedding stack. If
any test here starts needing one of those, the pure logic has leaked a
dependency and that is itself the bug.
"""

import pytest

from agents.assessment import (
    assess_age_risk,
    assess_symptoms,
    assess_vitals,
    combine_findings,
    derive_escalation,
    findings_from_extraction,
    symptom_severity,
    vital_severity_map,
)
from models import (
    ClinicalFinding,
    EscalationLevel,
    ExtractedSymptom,
    FindingCategory,
    FindingSeverity,
    Patient,
    SymptomStatus,
    VitalSigns,
)

CRITICAL = FindingSeverity.CRITICAL
CONCERNING = FindingSeverity.CONCERNING

NORMAL_VITALS = dict(
    heart_rate=78, systolic_bp=122, diastolic_bp=78,
    respiratory_rate=16, spo2=98.0, temperature_c=36.8,
)

# A resting HR/RR/BP that's normal for a toddler reads as abnormal under
# adult thresholds and vice versa — that's the whole point of age-banding.
# Pediatric-specific tests need their own "normal" baseline rather than
# overriding one field on the adult baseline above.
NORMAL_PEDIATRIC_VITALS = dict(
    heart_rate=110, systolic_bp=95, diastolic_bp=60,
    respiratory_rate=26, spo2=98.0, temperature_c=37.0,
)


def make_vitals(**overrides) -> VitalSigns:
    return VitalSigns(**{**NORMAL_VITALS, **overrides})


def make_pediatric_vitals(**overrides) -> VitalSigns:
    return VitalSigns(**{**NORMAL_PEDIATRIC_VITALS, **overrides})


def make_patient(complaint="routine follow up", age=40, **vital_overrides) -> Patient:
    return Patient(
        patient_id="PT-0001",
        name="Test Patient",
        age=age,
        weight_kg=70.0,
        chief_complaint=complaint,
        vitals=make_vitals(**vital_overrides),
    )


def finding(severity=CONCERNING, category=FindingCategory.VITAL, detail="x"):
    return ClinicalFinding(severity=severity, category=category, detail=detail)


# ─── Vital Thresholds ─────────────────────────────────────────────────────────

class TestVitalThresholds:
    """Adult age band — the default for everything in this class."""

    def test_normal_vitals_produce_no_findings(self):
        assert assess_vitals(make_vitals(), "adult") == []

    @pytest.mark.parametrize("field,value,expected", [
        # Heart rate
        ("heart_rate", 78, None),
        ("heart_rate", 101, CONCERNING),
        ("heart_rate", 131, CRITICAL),
        ("heart_rate", 59, CONCERNING),
        ("heart_rate", 44, CRITICAL),
        # Respiratory rate
        ("respiratory_rate", 21, CONCERNING),
        ("respiratory_rate", 31, CRITICAL),
        ("respiratory_rate", 7, CRITICAL),
        # Oxygen saturation
        ("spo2", 93.0, CONCERNING),
        ("spo2", 89.0, CRITICAL),
        ("spo2", 100.0, None),  # no upper bound on SpO2
        # Temperature
        ("temperature_c", 38.6, CONCERNING),
        ("temperature_c", 40.1, CRITICAL),
        ("temperature_c", 34.9, CRITICAL),
        # Systolic
        ("systolic_bp", 181, CONCERNING),
        ("systolic_bp", 221, CRITICAL),
        ("systolic_bp", 89, CRITICAL),
        # Diastolic — previously never evaluated at all
        ("diastolic_bp", 111, CONCERNING),
        ("diastolic_bp", 121, CRITICAL),
    ])
    def test_threshold_tiers(self, field, value, expected):
        findings = assess_vitals(make_vitals(**{field: value}), "adult")
        if expected is None:
            assert findings == []
        else:
            assert len(findings) == 1
            assert findings[0].severity is expected
            assert findings[0].category is FindingCategory.VITAL

    def test_boundary_values_are_not_findings(self):
        """Thresholds are exclusive: exactly at the bound is still in range."""
        assert assess_vitals(make_vitals(heart_rate=100), "adult") == []
        assert assess_vitals(make_vitals(heart_rate=60), "adult") == []
        assert assess_vitals(make_vitals(spo2=94.0), "adult") == []
        assert assess_vitals(make_vitals(temperature_c=38.5), "adult") == []

    def test_one_finding_per_vital_not_two(self):
        """A critically high value must not also report as concerning."""
        findings = assess_vitals(make_vitals(heart_rate=165), "adult")
        assert len(findings) == 1
        assert findings[0].severity is CRITICAL

    def test_dangerous_diastolic_is_caught_behind_normal_systolic(self):
        """Regression: 138/124 previously displayed as entirely normal."""
        findings = assess_vitals(make_vitals(systolic_bp=138, diastolic_bp=124), "adult")
        assert [f.severity for f in findings] == [CRITICAL]

    def test_severity_map_matches_findings(self):
        """The UI's highlighting must agree with the clinical rules."""
        vitals = make_vitals(heart_rate=165, spo2=92.0)
        severities = vital_severity_map(vitals, "adult")
        assert severities == {"heart_rate": CRITICAL, "spo2": CONCERNING}


class TestPediatricVitalThresholds:
    """
    Age-banded thresholds for age <= 5 (config.py's CRITICAL_VITALS_PEDIATRIC /
    CONCERNING_VITALS_PEDIATRIC — drafted, pending clinical review, see the
    note in config.py and evals/README.md's review pattern).
    """

    def test_the_documented_bug_is_fixed(self):
        """
        The exact case from CLAUDE.md / README's Known Limitations history:
        a well 3-year-old at HR 110 / RR 26 previously registered as both
        tachycardic and tachypneic against adult-only thresholds.
        """
        assert assess_vitals(make_pediatric_vitals(), "pediatric") == []

    def test_the_same_values_still_flag_for_an_adult(self):
        """Confirms this is genuinely age-banded, not just a raised ceiling
        that would also silently stop flagging adults: the same HR/RR that
        are normal for a toddler read as concerning for an adult."""
        vitals = make_vitals(heart_rate=110, respiratory_rate=26)
        findings = assess_vitals(vitals, "adult")
        assert len(findings) == 2
        assert {f.severity for f in findings} == {CONCERNING}

    def test_a_normal_pediatric_systolic_is_not_hypotensive(self):
        """A pediatric SBP of 95 is normal, not the adult 'borderline
        hypotension' (<100) it would wrongly register as under adult ranges."""
        assert assess_vitals(make_pediatric_vitals(), "pediatric") == []

    def test_pediatric_critical_tachycardia_still_flags(self):
        findings = assess_vitals(make_pediatric_vitals(heart_rate=190), "pediatric")
        assert len(findings) == 1
        assert findings[0].severity is CRITICAL

    def test_pediatric_hypotension_flags_at_a_lower_threshold_than_adult(self):
        """SBP 85: below the adult critical floor (90) but still normal for
        the pediatric band (concerning floor 80). Checked as a specific
        finding, not the whole list, since HR/RR normal-for-pediatric values
        read as abnormal under adult thresholds regardless of SBP."""
        vitals = make_pediatric_vitals(systolic_bp=85)
        adult_findings = assess_vitals(vitals, "adult")
        pediatric_findings = assess_vitals(vitals, "pediatric")
        assert any(f.severity is CRITICAL and "SBP" in f.detail for f in adult_findings)
        assert not any("SBP" in f.detail for f in pediatric_findings)


class TestGeriatricVitalThresholds:
    """Geriatric intentionally reuses the adult tables — see config.py."""

    def test_geriatric_uses_adult_thresholds(self):
        vitals = make_vitals(heart_rate=110)
        assert assess_vitals(vitals, "geriatric") == assess_vitals(vitals, "adult")


# ─── Symptom Extraction ───────────────────────────────────────────────────────

class TestSymptomExtraction:

    def test_present_symptom_becomes_a_finding(self):
        findings = findings_from_extraction(
            [ExtractedSymptom(symptom="chest pain", status=SymptomStatus.PRESENT)]
        )
        assert len(findings) == 1
        assert findings[0].severity is CONCERNING
        assert findings[0].category is FindingCategory.SYMPTOM

    @pytest.mark.parametrize("status", [SymptomStatus.DENIED, SymptomStatus.HISTORICAL])
    def test_denied_and_historical_produce_no_findings(self, status):
        """The entire point of extraction: 'denies chest pain' is not chest pain."""
        findings = findings_from_extraction(
            [ExtractedSymptom(symptom="chest pain", status=status)]
        )
        assert findings == []

    def test_mixed_statuses_keep_only_present(self):
        findings = findings_from_extraction([
            ExtractedSymptom(symptom="syncope", status=SymptomStatus.HISTORICAL),
            ExtractedSymptom(symptom="chest pressure", status=SymptomStatus.PRESENT),
            ExtractedSymptom(symptom="shortness of breath", status=SymptomStatus.DENIED),
        ])
        assert len(findings) == 1
        assert "chest pressure" in findings[0].detail

    def test_critical_symptom_maps_to_critical_severity(self):
        findings = findings_from_extraction(
            [ExtractedSymptom(symptom="unresponsive", status=SymptomStatus.PRESENT)]
        )
        assert findings[0].severity is CRITICAL

    def test_out_of_vocabulary_symptom_is_discarded(self):
        """A hallucinated term has no defined severity and must not become a finding."""
        findings = findings_from_extraction(
            [ExtractedSymptom(symptom="vague unease", status=SymptomStatus.PRESENT)]
        )
        assert findings == []

    def test_symptom_severity_is_case_insensitive(self):
        assert symptom_severity("CHEST PAIN") is CONCERNING
        assert symptom_severity("  Unresponsive  ") is CRITICAL
        assert symptom_severity("not a symptom") is None


# ─── Keyword Fallback ─────────────────────────────────────────────────────────

class TestKeywordFallback:
    """
    The substring scan used only when the LLM is unavailable. Its known
    weakness is asserted rather than hidden, so that if extraction ever
    silently stops running these tests document what behaviour returns.
    """

    def test_detects_a_present_symptom(self):
        findings = assess_symptoms("crushing chest pain radiating to jaw")
        assert any("chest pain" in f.detail for f in findings)

    def test_cannot_handle_negation(self):
        """Documents the limitation that structured extraction exists to fix."""
        findings = assess_symptoms("denies chest pain, here for medication refill")
        assert findings, "fallback is expected to false-positive on negation"

    def test_narrow_vocabulary_avoids_the_worst_false_positives(self):
        """Bare 'abdominal pain' and 'stroke' were removed from the vocabulary."""
        assert assess_symptoms("mild abdominal pain x 3 days, eating normally") == []
        assert assess_symptoms("history of stroke in 2019, here for suture removal") == []


# ─── Age Risk ─────────────────────────────────────────────────────────────────

class TestAgeRisk:

    @pytest.mark.parametrize("age", [3, 78])
    def test_age_alone_is_not_a_finding(self, age):
        assert assess_age_risk(make_patient(age=age), []) == []

    @pytest.mark.parametrize("age", [3, 78])
    def test_high_risk_age_with_another_finding_amplifies(self, age):
        findings = assess_age_risk(make_patient(age=age), [finding()])
        assert len(findings) == 1
        assert findings[0].category is FindingCategory.AGE

    def test_adult_age_never_amplifies(self):
        assert assess_age_risk(make_patient(age=40), [finding()]) == []

    def test_combine_applies_age_risk_to_the_merged_set(self):
        """Age risk must see symptom findings, which arrive after vitals."""
        patient = make_patient(age=78)
        symptom = finding(category=FindingCategory.SYMPTOM, detail="chest pain")
        combined = combine_findings(patient, [], [symptom])
        assert any(f.category is FindingCategory.AGE for f in combined)


# ─── Escalation Derivation ────────────────────────────────────────────────────

class TestEscalation:

    def test_no_score_no_findings_is_routine(self):
        level, reasons = derive_escalation(5, [])
        assert level is EscalationLevel.NONE
        assert reasons == []

    @pytest.mark.parametrize("esi", [1, 2])
    def test_low_esi_always_escalates_immediately(self, esi):
        level, reasons = derive_escalation(esi, [])
        assert level is EscalationLevel.IMMEDIATE
        assert reasons

    @pytest.mark.parametrize("esi", [3, 4, 5])
    def test_higher_esi_alone_does_not_escalate(self, esi):
        level, _ = derive_escalation(esi, [])
        assert level is EscalationLevel.NONE

    def test_critical_finding_escalates_even_at_low_acuity_score(self):
        """A critical vital must not be suppressed by a reassuring ESI."""
        level, _ = derive_escalation(5, [finding(severity=CRITICAL)])
        assert level is EscalationLevel.IMMEDIATE

    def test_concerning_finding_elevates_but_does_not_escalate(self):
        level, _ = derive_escalation(3, [finding(severity=CONCERNING)])
        assert level is EscalationLevel.ELEVATED

    def test_missing_score_with_no_findings_does_not_escalate_on_its_own(self):
        """A missing score is handled as a system error upstream, not here."""
        level, _ = derive_escalation(None, [])
        assert level is EscalationLevel.NONE

    def test_reasons_name_the_critical_findings(self):
        crit = finding(severity=CRITICAL, detail="Significant hypoxia — SpO2 88%")
        _, reasons = derive_escalation(3, [crit])
        assert any("hypoxia" in r for r in reasons)


# ─── End-to-end deterministic path ────────────────────────────────────────────

class TestDeterministicPath:
    """
    Full assess -> escalate against realistic presentations, using the
    extraction results a correct model would return.
    """

    def _escalate(self, patient, extraction, esi):
        findings = combine_findings(
            patient,
            assess_vitals(patient.vitals, patient.age_group),
            findings_from_extraction(extraction),
        )
        return derive_escalation(esi, findings)[0]

    def test_medication_refill_denying_chest_pain_is_routine(self):
        """Regression: previously flagged, and skipped scoring entirely."""
        level = self._escalate(
            make_patient("denies chest pain, here for medication refill"),
            [ExtractedSymptom(symptom="chest pain", status=SymptomStatus.DENIED)],
            esi=5,
        )
        assert level is EscalationLevel.NONE

    def test_suture_removal_with_stroke_history_is_routine(self):
        level = self._escalate(
            make_patient("history of stroke in 2019, here for suture removal"),
            [],
            esi=5,
        )
        assert level is EscalationLevel.NONE

    def test_septic_presentation_escalates_immediately(self):
        patient = make_patient(
            "fever and confusion since yesterday", age=78,
            heart_rate=134, respiratory_rate=32, spo2=89.0, temperature_c=39.4,
        )
        level = self._escalate(patient, [], esi=2)
        assert level is EscalationLevel.IMMEDIATE

    def test_asymptomatic_hypertension_elevates_only(self):
        patient = make_patient("medication refill", systolic_bp=186, diastolic_bp=104)
        level = self._escalate(patient, [], esi=3)
        assert level is EscalationLevel.ELEVATED
