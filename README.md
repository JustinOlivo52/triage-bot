# Triage Bot — ED Triage RAG AI Assistant

An AI-powered emergency department triage assistant built with LangGraph, LangChain, and Claude. Combines retrieval-augmented generation (RAG) grounded in clinical guidelines with a deterministic multi-layer red flag system to assist triage nurses and physicians with ESI scoring and high-risk patient identification.

> **Clinical Disclaimer:** This tool is a decision-support prototype only. It does not replace clinical judgment or qualified medical personnel. All triage decisions must be validated by a licensed clinician.

---

## What It Does

- Accepts patient intake (name, age, weight, chief complaint, vitals)
- Runs a **3-layer red flag evaluation** to immediately surface high-risk patients to the physician
- For non-flagged patients, performs **LLM-driven ESI 1–5 triage scoring** grounded in clinical guidelines via RAG
- Maintains a **persistent multi-patient queue** with unique patient IDs (PT-0001 format) and returning patient detection across sessions
- Displays a structured **PatientCard** with vitals, ESI score, clinical reasoning, and recommended interventions

---

## Red Flag System

The red flag system uses three sequential evaluation layers — designed to catch critical patients through both hard rules and holistic clinical reasoning:

| Layer | Type | Trigger |
|-------|------|---------|
| **Layer 1** | Hard Rule (embedded in Layer 3) | ESI score 1 or 2 always flags |
| **Layer 2** | Deterministic | Out-of-range vitals, high-risk symptom keywords, age amplifiers (pediatric ≤5, geriatric ≥65) |
| **Layer 3** | LLM Reasoning | Claude evaluates the full clinical picture holistically via RAG — catches edge cases Layer 2 misses |

When a red flag fires, the physician receives an immediate structured alert with urgency statement, trigger reasons, abnormal vitals, clinical concerns, and recommended actions.

**Fail-safe:** If the LLM errors at Layer 3, the patient is flagged rather than cleared — always safer to over-triage.

---

## Architecture

```
Streamlit UI
    │
    ▼
PatientStore (JSON persistence)
    │
    ▼
LangGraph StateGraph
    ├── red_flag_node  ──── Layer 2 (deterministic) + Layer 3 (LLM + RAG)
    │       │
    │  [flagged?]
    │    YES │ NO
    │       ▼  ▼
    │  build_card_node   triage_node (LLM ESI scoring + RAG)
    │                         │
    │                         ▼
    └──────────────── build_card_node
                           │
                           ▼
                       PatientCard
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
│   ├── red_flag.py          # 3-layer red flag evaluator
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

- **Persistent patient queue** — survives app restarts; JSON-backed storage with atomic read/write
- **Returning patient detection** — flags patients with prior visits by name matching across sessions
- **Structured LLM output** — Pydantic v2 schema-enforced responses via `with_structured_output()` — no hallucinated JSON fields
- **Deterministic clinical decisions** — `temperature=0` on all LLM calls for consistent, reproducible triage
- **Abnormal vital highlighting** — UI color-codes out-of-range vitals against ESI v4 danger zone thresholds
- **Physician alert banners** — red flagged patients surface at the top of every page until resolved
- **Cost-efficient embeddings** — local HuggingFace embeddings mean zero embedding API cost

---

## V2 Roadmap

- [ ] Load and index clinical guideline PDFs into ChromaDB
- [ ] `.env.example` and setup documentation
- [ ] Shift handoff report generation
- [ ] Nurse annotation / override workflow
- [ ] Audit log for all triage decisions
- [ ] Unit and integration test suite

---

## Background

Built by an AI Engineer with 9 years of clinical emergency department experience. The red flag criteria, vital thresholds, and ESI scoring logic reflect real-world ED triage practice and ESI v4 guidelines.
