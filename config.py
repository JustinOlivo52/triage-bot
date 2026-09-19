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
# Both tiers are adult values. Pediatric and geriatric ranges differ materially
# (a well 3-year-old sits around HR 110 / RR 26) and are handled separately.

CRITICAL_VITALS: dict = {
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

CONCERNING_VITALS: dict = {
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

AGE_THRESHOLDS: dict = {
    "pediatric_max": 5,    # age ≤ 5 — atypical presentations, adult ranges do not apply
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
