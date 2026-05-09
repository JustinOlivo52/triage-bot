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
CHROMA_DIR = BASE_DIR / "chroma_db"

# ─── Anthropic ────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL: str = "claude-sonnet-4-6"

# ─── Embeddings ───────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ─── ChromaDB ─────────────────────────────────────────────────────────────────

CHROMA_COLLECTION_NAME: str = "triage_knowledge"

# ─── RAG Settings ─────────────────────────────────────────────────────────────

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
RETRIEVAL_K: int = 5

# ─── Clinical Thresholds — Layer 1 (Hard ESI Rule) ────────────────────────────

ESI_RED_FLAG_MAX: int = 2  # ESI 1 or 2 always triggers red flag

# ─── Clinical Thresholds — Layer 2 (Deterministic Risk Factors) ───────────────

VITAL_THRESHOLDS: dict = {
    "hr_high": 100,        # bpm — tachycardia
    "hr_low": 60,          # bpm — bradycardia
    "rr_high": 20,         # breaths/min — tachypnea
    "spo2_low": 94,        # % — hypoxia threshold
    "temp_high_c": 38.5,   # °C — fever
    "temp_low_c": 36.0,    # °C — hypothermia
    "sbp_high": 180,       # mmHg — hypertensive urgency
    "sbp_low": 90,         # mmHg — hypotension
}

AGE_THRESHOLDS: dict = {
    "pediatric_max": 5,    # age ≤ 5 flagged with any acute symptom
    "geriatric_min": 65,   # age ≥ 65 flagged with any concerning symptom
}

# Symptoms that trigger an immediate Layer 2 flag regardless of ESI score.
# Layer 3 LLM reasoning catches edge cases not covered here.
RED_FLAG_SYMPTOMS: list[str] = [
    "chest pain",
    "chest tightness",
    "chest pressure",
    "shortness of breath",
    "difficulty breathing",
    "can't breathe",
    "severe abdominal pain",
    "abdominal pain",
    "severe nausea",
    "severe vomiting",
    "unresponsive",
    "loss of consciousness",
    "syncope",
    "altered mental status",
    "confusion",
    "stroke",
    "facial droop",
    "arm weakness",
    "seizure",
    "anaphylaxis",
    "allergic reaction",
    "severe bleeding",
    "hemoptysis",
]

# ─── Patient ID ───────────────────────────────────────────────────────────────

PATIENT_ID_PREFIX: str = "PT"  # e.g. PT-0042
