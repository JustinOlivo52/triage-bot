# Triage Bot — ED Triage RAG AI Assistant

An AI-powered emergency department triage assistant built with LangGraph, LangChain, and Claude. Combines retrieval-augmented generation (RAG) grounded in clinical guidelines with a deterministic multi-layer red flag system to assist triage nurses and physicians with ESI scoring and high-risk patient identification.

> **Clinical Disclaimer:** This tool is a decision-support prototype only. It does not replace clinical judgment or qualified medical personnel. All triage decisions must be validated by a licensed clinician.

---

## What It Does

- Accepts patient intake (name, age, weight, chief complaint, vitals)
- Runs a **deterministic clinical assessment** (vitals, symptoms, age risk) with no LLM cost
- Assigns an **ESI 1–5 score to every patient** via LLM reasoning grounded in clinical guidelines through RAG
- Derives an **escalation level** from that score plus the deterministic findings, and alerts the physician when it is immediate
- Maintains a **persistent multi-patient queue** with unique patient IDs (PT-0001 format) and returning patient detection across sessions
- Displays a structured **PatientCard** with vitals, ESI score, escalation status, clinical reasoning, and recommended interventions

---

## Scoring and Escalation

Acuity and urgency are one clinical judgment, not two competing branches. Every patient is scored, and escalation is layered on top of that score rather than replacing it.

```
assess (deterministic)  →  triage (ESI 1–5)  →  escalate (derived)  →  patient card
```

**Findings** come from the deterministic pass and carry a severity, so `HR 104` and `HR 165` are not treated as the same event:

| Severity | Meaning | Example |
|----------|---------|---------|
| `CRITICAL` | Warrants a physician now, independent of ESI | HR > 130, SpO2 < 90, SBP < 90 |
| `CONCERNING` | Outside normal range, raises concern | HR > 100, SpO2 < 94, temp > 38.5 °C |

**Escalation** is a pure function of the ESI score and those findings:

| Level | Rule |
|-------|------|
| `IMMEDIATE` | ESI ≤ 2, **or** any critical finding |
| `ELEVATED` | Any remaining finding |
| `NONE` | Scored, nothing outstanding |

Only `IMMEDIATE` triggers the physician-summary LLM call, so routine and elevated patients cost nothing beyond their scoring.

**Fail-safe:** if triage reasoning fails, the patient is escalated to `IMMEDIATE` rather than cleared — but no ESI score is fabricated, and the card is marked as a *system error* so a pipeline failure is never mistaken for a clinical judgment.

Age is not a finding on its own. Pediatric (≤5) and geriatric (≥65) patients are flagged only when another abnormal finding is already present, because both ends of the age range blunt the usual warning signs.

---

## Architecture

```
Streamlit UI
    │
    ▼
PatientStore (JSON persistence)
    │
    ▼
LangGraph StateGraph  —  linear, every patient traverses every node
    │
    ├── assess_node      deterministic findings (vitals, symptoms, age) — no LLM
    │        │
    ├── triage_node      ESI 1–5 scoring (LLM + RAG), sees the findings
    │        │
    ├── escalate_node    derive escalation; physician summary if IMMEDIATE
    │        │
    └── build_card_node
             │
             ▼
         PatientCard  (ESI score + escalation level, always both)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Orchestration | LangGraph + LangChain |
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| RAG / Vector Store | ChromaDB (local, persistent) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (CPU, no API cost) |
| Document Loading | PyMuPDF (`fitz`) |
| Data Validation | Pydantic v2 |
| Persistence | JSON flat-file patient store |

---

## Project Structure

```
triage-bot/
├── main.py                  # Streamlit UI
├── config.py                # Central config, clinical thresholds, model settings
├── models.py                # Pydantic v2 data models + LangGraph TypedDict state
├── requirements.txt
├── .env.example             # Environment variable template
│
├── agents/
│   ├── assessment.py        # Deterministic clinical rules + escalation logic (pure, no LLM)
│   ├── escalation.py        # Escalation assembly + physician alert generation
│   └── triage_agent.py      # LangGraph StateGraph + triage node
│
├── rag/
│   ├── ingest.py            # PDF loading, chunking, embedding, ChromaDB ingestion
│   └── retriever.py         # Similarity search and context formatting
│
├── memory/
│   └── patient_store.py     # JSON-backed persistent patient queue
│
└── data/                    # Place clinical guideline PDFs here (gitignored)
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/triage-bot.git
cd triage-bot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your Anthropic API key to .env
```

### 3. Add clinical guidelines (optional but recommended)

Place PDF clinical guidelines (ESI guidelines, clinical protocols, etc.) in the `/data` folder, then run:

```bash
python -m rag.ingest
```

The app will run without PDFs — the LLM will rely on its training knowledge — but RAG grounding significantly improves triage accuracy.

### 4. Launch

```bash
streamlit run main.py
```

---

## Key Features

- **Every patient gets a score** — escalation is reported alongside the ESI level, never instead of it
- **Testable clinical rules** — all threshold and escalation logic lives in `agents/assessment.py` as pure functions with no LLM or network dependency
- **Single source of truth for thresholds** — the triage prompt renders its vital-sign ranges from `config.py`, so the prompt cannot drift from the code
- **Two-tier severity** — critical and concerning findings are distinguished rather than collapsed into one binary flag
- **Cost-aware escalation** — only `IMMEDIATE` patients trigger a physician-summary LLM call
- **Structured LLM output** — Pydantic v2 schema-enforced responses via `with_structured_output()`, no hallucinated JSON fields
- **Abnormal vital highlighting** — the UI reads the same thresholds the clinical rules use, and marks severity with text as well as colour so the signal survives in greyscale
- **Honest failure modes** — a pipeline failure is surfaced as a system error, not disguised as a clinical alert or a fabricated ESI 3
- **Runs ungrounded** — with no guideline PDFs the app degrades to unretrieved reasoning instead of failing to start
- **Persistent patient queue** — survives app restarts, JSON-backed
- **Returning patient detection** — flags patients with prior visits by name matching across sessions
- **Cost-efficient embeddings** — local HuggingFace embeddings mean zero embedding API cost

---

## V2 Roadmap

**Done**
- [x] Separate ESI scoring from escalation so every patient is scored
- [x] Two-tier finding severity (critical vs concerning)
- [x] Pure, dependency-free clinical rules module
- [x] System failures distinguished from clinical alerts
- [x] `.env.example` and setup documentation

**Next**
- [ ] Structured symptom extraction to replace keyword matching (handles negation and history)
- [ ] Unit and integration test suite
- [ ] Eval set of clinician-scored vignettes with a measured agreement rate
- [ ] Atomic writes and a single cached read per render in the patient store
- [ ] Nurse annotation / override workflow
- [ ] Age-banded vital thresholds (current ranges are adult values)
- [ ] Audit log for all triage decisions
- [ ] Shift handoff report generation
- [ ] Load and index clinical guideline PDFs into ChromaDB

---

## Known Limitations

- **Symptom matching is a substring scan.** It cannot distinguish "chest pain" from "denies chest pain" or "history of chest pain". The keyword lists are kept narrow to limit false positives, and the triage prompt instructs the model to disregard findings the complaint does not support, but the scan itself is still naive. Structured extraction is the next item on the roadmap.
- **Vital thresholds are adult values.** They are applied to all ages. A well 3-year-old sits around HR 110 / RR 26 and will register as tachycardic and tachypneic.
- **Writes are not atomic.** A crash mid-write can corrupt the patient store.
- **No accuracy measurement yet.** There is no eval set, so the system's agreement with expert ESI assignment is currently unknown.

---

## Background

Built by an AI Engineer with 9 years of clinical emergency department experience. The red flag criteria, vital thresholds, and ESI scoring logic reflect real-world ED triage practice and ESI v4 guidelines.
