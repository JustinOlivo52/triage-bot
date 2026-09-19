"""
config.py — Central configuration for triage-bot.
Loads environment variables, defines model settings,
clinical thresholds, and RAG parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# The clinical reference and its precomputed embedding index are both committed
# to the repo, so the app never builds an index at runtime. Rebuild with
# `python -m scripts.build_index` after editing the reference.
REFERENCE_DOC = DATA_DIR / "esi_reference.md"
REFERENCE_INDEX = DATA_DIR / "reference_index.json"
SEED_COHORT = DATA_DIR / "seed_cohort.json"

# ─── Anthropic ────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Two models, because the two call sites have different requirements.
#
# TRIAGE_MODEL performs the actual clinical judgment: reading the complaint,
# weighing it against the vitals, and assigning an ESI level. Under-triage is
# the dangerous failure mode here, so this call gets the more capable model.
#
# SUMMARY_MODEL writes the physician alert narrative. By the time it runs, the
# escalation decision and the ESI score are already settled by the rules and
# the triage call — it is composing prose from established facts.
#
# Which model is *correct* for each role is not yet measured. These are
# reasoned defaults, not validated ones; the eval harness is what will turn
# them into an evidence-based choice.
TRIAGE_MODEL: str = "claude-opus-5"
SUMMARY_MODEL: str = "claude-sonnet-5"

# ─── Embeddings ───────────────────────────────────────────────────────────────
#
# Anthropic ships no embedding model; Voyage is its documented partner. Query
# embedding is one small HTTP call, so there is no local model and no torch.
#
# Retrieval degrades in two steps rather than failing:
#   key present  → semantic search over the precomputed index
#   key absent   → lexical overlap scoring over the same index
#   index absent → no context; triage proceeds ungrounded
#
# That means the app needs exactly one secret (ANTHROPIC_API_KEY) to be fully
# functional, and the embeddings key is an upgrade rather than a requirement.

VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
EMBEDDING_MODEL: str = "voyage-3.5-lite"
EMBEDDING_ENDPOINT: str = "https://api.voyageai.com/v1/embeddings"
EMBEDDING_TIMEOUT_SECONDS: float = 10.0

# ─── Retrieval Settings ───────────────────────────────────────────────────────

RETRIEVAL_K: int = 4
MAX_CHUNK_CHARS: int = 1200  # sections longer than this are split further

# ─── Demo Mode ────────────────────────────────────────────────────────────────
#
# Set DEMO_MODE=true for a public deployment. Patients are then scoped to the
# visitor's browser session rather than a shared file, so visitors cannot see
# or wipe each other's queues, and live triage runs are capped.

DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").strip().lower() in {"1", "true", "yes"}
LIVE_TRIAGE_LIMIT: int = int(os.getenv("LIVE_TRIAGE_LIMIT", "3"))

# ─── Escalation Thresholds ────────────────────────────────────────────────────

# An ESI at or below this level always warrants immediate physician attention.
ESI_IMMEDIATE_MAX: int = 2

# ─── Clinical Thresholds — Vital Signs ────────────────────────────────────────
#
# Two tiers, because "abnormal" and "dangerous" are not the same thing.
#
#   CRITICAL   — derangement that warrants a physician now, independent of ESI.
#   CONCERNING — outside normal range; raises concern but does not by itself
#                mean the patient cannot wait.
#
# Age-banded: pediatric (age <= 5, matching AGE_THRESHOLDS below) gets its own
# tables, because a well 3-year-old sits around HR 110 / RR 26 — both flagged
# abnormal by adult ranges, which was a real bug (see CLAUDE.md / README Known
# Limitations history). Geriatric patients keep the adult tables: their vital
# signs aren't shifted the way pediatric ones are, and the actual clinical
# concern for that age band — a blunted response can mask severity even when
# numbers look normal — is handled by assess_age_risk() amplifying whatever
# finding IS present, not by moving the raw thresholds themselves.
#
# CRITICAL_VITALS_PEDIATRIC / CONCERNING_VITALS_PEDIATRIC are drafted from
# general pediatric vital-sign norms for this age band, the same way
# data/esi_reference.md and evals/vignettes.json were drafted — pending
# Justin's clinical review before being treated as fully validated (see the
# note in data/esi_reference.md's Vital Sign Danger Zones section).

CRITICAL_VITALS_ADULT: dict = {
    "hr_high": 130,        # bpm — marked tachycardia
    "hr_low": 45,          # bpm — marked bradycardia
    "rr_high": 30,         # breaths/min — severe tachypnea
    "rr_low": 8,           # breaths/min — respiratory depression
    "spo2_low": 90,        # % — significant hypoxia
    "temp_high": 40.0,     # °C — hyperpyrexia
    "temp_low": 35.0,      # °C — hypothermia
    "sbp_high": 220,       # mmHg — severe hypertension
    "sbp_low": 90,         # mmHg — hypotension
    "dbp_high": 120,       # mmHg — severe diastolic hypertension
}

CONCERNING_VITALS_ADULT: dict = {
    "hr_high": 100,        # bpm — tachycardia
    "hr_low": 60,          # bpm — bradycardia
    "rr_high": 20,         # breaths/min — tachypnea
    "rr_low": 12,          # breaths/min — low-normal respiratory rate
    "spo2_low": 94,        # % — borderline oxygenation
    "temp_high": 38.5,     # °C — fever
    "temp_low": 36.0,      # °C — low temperature
    "sbp_high": 180,       # mmHg — hypertensive urgency
    "sbp_low": 100,        # mmHg — borderline hypotension
    "dbp_high": 110,       # mmHg — diastolic hypertension
}

# DRAFT — pending clinical review. Fever/hypoxia/hypothermia cutoffs are kept
# the same as adult (those definitions don't meaningfully shift for this age
# band); heart rate, respiratory rate, and blood pressure are lower than
# adult, reflecting normal pediatric physiology.
CRITICAL_VITALS_PEDIATRIC: dict = {
    "hr_high": 180,        # bpm — marked tachycardia for this age band
    "hr_low": 70,          # bpm — marked bradycardia
    "rr_high": 50,         # breaths/min — severe tachypnea
    "rr_low": 15,          # breaths/min — respiratory depression
    "spo2_low": 90,        # % — significant hypoxia
    "temp_high": 40.0,     # °C — hyperpyrexia
    "temp_low": 35.0,      # °C — hypothermia
    "sbp_high": 140,       # mmHg — severe hypertension (rare at this age, but a ceiling)
    "sbp_low": 70,         # mmHg — hypotension
    "dbp_high": 90,        # mmHg — severe diastolic hypertension
}

# DRAFT — pending clinical review. See CRITICAL_VITALS_PEDIATRIC above.
CONCERNING_VITALS_PEDIATRIC: dict = {
    "hr_high": 140,        # bpm — tachycardia
    "hr_low": 80,          # bpm — bradycardia
    "rr_high": 30,         # breaths/min — tachypnea
    "rr_low": 20,          # breaths/min — low-normal respiratory rate
    "spo2_low": 95,        # % — borderline oxygenation
    "temp_high": 38.5,     # °C — fever
    "temp_low": 36.0,      # °C — low temperature
    "sbp_high": 120,       # mmHg — hypertensive urgency
    "sbp_low": 80,         # mmHg — borderline hypotension
    "dbp_high": 80,        # mmHg — diastolic hypertension
}

# Keyed by Patient.age_group (models.py) — "pediatric" / "adult" / "geriatric".
# Geriatric intentionally maps to the adult tables; see the note above.
CRITICAL_VITALS_BY_AGE_GROUP: dict[str, dict] = {
    "pediatric": CRITICAL_VITALS_PEDIATRIC,
    "adult": CRITICAL_VITALS_ADULT,
    "geriatric": CRITICAL_VITALS_ADULT,
}
CONCERNING_VITALS_BY_AGE_GROUP: dict[str, dict] = {
    "pediatric": CONCERNING_VITALS_PEDIATRIC,
    "adult": CONCERNING_VITALS_ADULT,
    "geriatric": CONCERNING_VITALS_ADULT,
}

AGE_THRESHOLDS: dict = {
    "pediatric_max": 5,    # age ≤ 5 — own vital-sign tables (above) + age-risk amplification
    "geriatric_min": 65,   # age ≥ 65 — blunted physiologic response masks severity
}

# ─── Clinical Thresholds — Symptoms ───────────────────────────────────────────
#
# Same two tiers. Deliberately narrow: these are matched against free text, so a
# broad term like "abdominal pain" matches most of the department and tells you
# nothing. Nuance is the LLM's job, not the keyword list's.

CRITICAL_SYMPTOMS: list[str] = [
    "unresponsive",
    "not breathing",
    "cardiac arrest",
    "anaphylaxis",
    "facial droop",
    "arm weakness",
    "slurred speech",
    "active seizure",
    "severe bleeding",
    "hemoptysis",
]

CONCERNING_SYMPTOMS: list[str] = [
    "chest pain",
    "chest tightness",
    "chest pressure",
    "shortness of breath",
    "difficulty breathing",
    "can't breathe",
    "severe abdominal pain",
    "loss of consciousness",
    "altered mental status",
    "syncope",
    "seizure",
    "allergic reaction",
    "severe headache",
    "suicidal",
]

# ─── Patient ID ───────────────────────────────────────────────────────────────

PATIENT_ID_PREFIX: str = "PT"  # e.g. PT-0042
