# Triage Bot — ED Triage Assistant

An AI-powered emergency department triage assistant: a FastAPI backend with a
real, role-gated auth model and a FHIR-shaped data model, a Streamlit UI that
talks to it over HTTP, and a LangGraph pipeline that combines deterministic
clinical rules with retrieval-augmented Claude reasoning for ESI scoring and
high-risk patient identification.

> **Clinical Disclaimer:** This tool is a decision-support prototype only. It does not replace clinical judgment or qualified medical personnel. All triage decisions must be validated by a licensed clinician.

---

## What It Does

- Nurse/physician/admin accounts, JWT-authenticated, backed by a role check on every route — not a public toy demo
- Accepts patient intake (name, age, weight, chief complaint, vitals) and runs a **deterministic clinical assessment** (vitals, symptoms, age risk) with no LLM cost
- Assigns an **ESI 1–5 score to every patient** via LLM reasoning grounded in clinical guidelines through RAG
- Derives an **escalation level** from that score plus the deterministic findings, and alerts the physician when it is immediate
- Persists every check-in as **FHIR-shaped records** (`Patient`, `Encounter`, `Observation`, `RiskAssessment`) in a real database, not a JSON file
- Lets a physician **resolve/dispose** an encounter, and lets an admin read the **append-only audit log** of who did what
- Displays a structured **PatientCard** with vitals, ESI score, escalation status, clinical reasoning, and recommended interventions

---

## Scoring and Escalation

Acuity and urgency are one clinical judgment, not two competing branches. Every patient is scored, and escalation is layered on top of that score rather than replacing it. This part of the system is unchanged since it was hardened in V1 — the V2 work below wraps it in a real service, it doesn't touch the clinical logic itself.

```
assess (deterministic)  →  triage (ESI 1–5)  →  escalate (derived)  →  patient card
```

**Findings** come from the deterministic pass and carry a severity, so `HR 104` and `HR 165` are not treated as the same event:

| Severity | Meaning | Example |
|----------|---------|---------|
| `CRITICAL` | Warrants a physician now, independent of ESI | HR > 130, SpO2 < 90, SBP < 90 |
| `CONCERNING` | Outside normal range, raises concern | HR > 100, SpO2 < 94, temp > 38.5 °C |

**Symptoms** are extracted, not keyword-matched. The triage call reports the status of each vocabulary term the complaint mentions, so `denies chest pain` and `history of stroke in 2019` produce no finding:

| Status | Becomes a finding? |
|--------|--------------------|
| `present` | Yes |
| `denied` | No — recorded for audit only |
| `historical` | No — recorded for audit only |

Extraction rides on the existing triage call rather than taking one of its own, so it costs no extra request. The vocabulary is supplied from `config.py` and severity is mapped on our side, which keeps the mapping deterministic and stops an out-of-vocabulary hallucination from becoming a finding of unknown weight.

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

Two services. Streamlit is a thin HTTP client — it holds no business logic, no storage, and imports nothing from `agents/` or `memory/`. Every action, including reading the queue, is a real API call carrying a JWT.

```
┌──────────────┐   HTTP + JWT    ┌──────────────────────────────────┐
│  Streamlit   │ ──────────────► │           FastAPI backend         │
│  (main.py)   │ ◄────────────── │                                    │
│              │                 │  api/routes/  auth · encounters ·  │
│ api_client.py│                 │               audit                │
└──────────────┘                 │  api/deps.py  get_current_user,    │
                                  │               require_role(...)   │
                                  │                                    │
                                  │  services/                        │
                                  │   ├─ triage_service ──────────────┼──► agents/triage_agent.py
                                  │   │   (FHIR persistence boundary)  │    (LangGraph pipeline,
                                  │   ├─ audit_service                 │     unchanged — no FHIR
                                  │   └─ demo_seed                     │     awareness at all)
                                  │                                    │
                                  │  db/  SQLAlchemy + Alembic         │
                                  └────────────────┬───────────────────┘
                                                   │
                                  ┌────────────────▼───────────────────┐
                                  │      Postgres (prod) / SQLite       │
                                  │  users · patients · encounters ·    │
                                  │  observations · risk_assessments ·  │
                                  │  audit_logs                         │
                                  └──────────────────────────────────────┘
```

Inside `agents/triage_agent.py`, the pipeline itself is still the same linear graph:

```
assess_node → triage_node → escalate_node → build_card_node
```

- `assess_node` — vitals only, against `config.py` thresholds. No LLM.
- `triage_node` — the LLM call. Assigns ESI 1–5 **and** extracts symptom status from the chief complaint in the same call.
- `escalate_node` — combines vital + symptom findings, applies age risk, derives escalation, generates a physician summary only if `IMMEDIATE`.
- `build_card_node` — assembles the `PatientCard` returned to the API layer.

`backend/services/triage_service.py` is the only place that knows both languages: it calls `run_triage()` unchanged, then maps the resulting `PatientCard` onto the FHIR-shaped rows below. The pipeline has zero awareness that FHIR, a database, or an API even exist.

---

## Data Model: FHIR-Shaped, Not FHIR-Certified

Real FHIR resource names and field semantics, mapped onto what the app actually collects — a legitimate signal to anyone who knows EHR data (Epic, Cerner, any HL7-conformant shop), but **not** a certified FHIR server: no terminology binding, no `$validate`, no complete resource set.

| Our concept | FHIR resource | Why this one |
|---|---|---|
| A person | `Patient` | identifier, name, birthDate, gender |
| One ED visit | `Encounter` | status, class=`EMER`, period, reasonCode (chief complaint) |
| One vital sign reading | `Observation` | code (LOINC), valueQuantity, effectiveDateTime — **one row per vital**, not one blob, because that's how a real EHR stores it |
| ESI score + escalation | `RiskAssessment` | subject, encounter, basis, prediction.outcome, prediction.qualitativeRisk, rationale — the actual FHIR resource for "here is an acuity/risk judgment about this patient" |

Every simplification is stated in the code, not left silent: `full_name` is one field, not FHIR's structured `HumanName`; `birth_date` is estimated from age at check-in (`birth_date_is_estimated` says so explicitly); `Encounter.reasonCode` is free text rather than a coded `CodeableConcept`; weight is modeled as an `Observation` (LOINC 29463-7), not a `Patient` field, matching real FHIR. `RiskAssessment.qualitativeRisk` uses FHIR's own risk-probability value set (`negligible | low | moderate | high | certain`), mapped from our escalation levels rather than inventing a parallel vocabulary.

`Encounter.card_json` caches the pipeline's native `PatientCard` output (recommended interventions, extracted symptoms, physician summary) alongside the FHIR rows — reconstructing that structure from `RiskAssessment.basis`/`.rationale` text alone would mean re-parsing free text back into structure. The FHIR rows stay the source of truth for anything that queries by LOINC code or risk level; the cache is what the UI actually reads.

---

## Auth & Roles

JWT bearer tokens (PyJWT), bcrypt password hashing, three roles enforced by a FastAPI dependency (`require_role(...)`) rather than scattered `if` checks in route bodies:

| Role | Can do |
|---|---|
| `nurse` | Check in patients, run triage, read the queue, search |
| `physician` | Everything a nurse can, **+** resolve/dispose an encounter |
| `admin` | Everything a physician can, **+** read the audit log, create accounts, end-of-shift reset |

A role hierarchy ("physician can do what a nurse can") is expressed by listing every role a route accepts, not by ordinal comparison — explicit over implicit, since that's not a fact the codebase should quietly assume everywhere.

**Demo mode** (`DEMO_MODE=true`): seeds three fixed accounts on startup — `demo_nurse` / `demo_physician` / `demo_admin`, all password `demo1234` (`DEMO_ACCOUNT_PASSWORD`) — plus the committed patient cohort (`data/seed_cohort.json`) into one real, shared department queue. Everyone who logs in, including a recruiter clicking through, sees the same thing real staff would see. Seeding is idempotent: safe on every restart, and it never overwrites real check-ins.

---

## Audit Log

Every mutating action writes one append-only row: `actor_user_id`, `action`, `resource_type`, `resource_id`, `timestamp`, JSON `metadata`. Covered today: login, account creation, patient check-in, triage run (and triage system-error), escalation resolution, and the end-of-shift reset. Append-only is enforced at the application layer — `GET /audit` (admin-only) is the only audit route; there is no update or delete endpoint, and a reset never deletes audit rows, including the row recording the reset itself. (A future hardening pass could add a database-level rule blocking `UPDATE`/`DELETE` on the table too — noted as a follow-up, not done here, since SQLite in the test suite has no equivalent mechanism to verify it against.)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI, Uvicorn |
| Auth | PyJWT, bcrypt |
| Database | SQLAlchemy 2.0 + Alembic — Postgres in production, SQLite for local dev and tests |
| UI | Streamlit (thin API client — `requests`) |
| Orchestration | LangGraph + LangChain |
| LLM | Anthropic Claude — `claude-opus-5` for ESI scoring, `claude-sonnet-5` for alert narrative |
| Retrieval | Precomputed JSON index (`data/reference_index.json`), numpy cosine similarity — Voyage AI embeddings when a key is present, token-overlap lexical search otherwise, no embeddings at all as the last fallback |
| Data Validation | Pydantic v2 |

---

## Project Structure

```
triage-bot/
├── main.py                  # Streamlit UI — thin client of backend/'s API
├── api_client.py            # HTTP wrapper main.py calls instead of importing agents/ or memory/
├── config.py                # Pipeline config: models, clinical thresholds, retrieval settings
├── models.py                # Pydantic v2 domain models + LangGraph TypedDict state
├── requirements.txt         # Streamlit + pipeline dependencies (~531MB footprint)
│
├── agents/
│   ├── assessment.py        # Deterministic clinical rules (pure, no LLM)
│   ├── escalation.py        # Escalation assembly + physician alert generation
│   ├── triage_agent.py      # LangGraph StateGraph — assess/triage/escalate/build_card
│   └── schema_utils.py      # Defensive coercion for structured LLM output
│
├── rag/
│   ├── ingest.py             # Markdown → chunk → embed → JSON index (build-time)
│   ├── embeddings.py         # Voyage API calls
│   └── retriever.py          # Cosine/lexical search, formatted context
│
├── memory/                  # Superseded by backend/ for the running app;
│   ├── backends.py          #  kept for local/offline use of the pipeline directly
│   └── patient_store.py     #  and for load_seed_cards(), reused by backend/services/demo_seed.py
│
├── backend/                  # The real service
│   ├── main.py                # FastAPI app, CORS, demo-seed startup hook, /health
│   ├── core/
│   │   ├── config.py           # Service settings — DB URL, JWT secret, CORS, demo mode
│   │   └── security.py         # Password hashing, JWT issue/verify
│   ├── db/                    # SQLAlchemy engine/session + declarative base
│   ├── models/                 # User, Patient, Encounter, Observation, RiskAssessment, AuditLog
│   ├── schemas/                 # Pydantic request/response shapes for the API
│   ├── services/
│   │   ├── triage_service.py    # The FHIR persistence boundary — check-in/triage/resolve/reset
│   │   ├── audit_service.py     # Read-only audit trail queries
│   │   └── demo_seed.py         # Fixed demo accounts + seed cohort, idempotent
│   ├── api/
│   │   ├── deps.py              # get_current_user, require_role(...)
│   │   └── routes/              # auth.py, encounters.py, audit.py
│   ├── alembic/                 # Migrations — the record of how the schema evolved
│   └── requirements.txt         # Backend-only deps, kept separate from the top-level file
│
├── data/                     # Clinical reference, precomputed index, seed cohort — committed
│
├── evals/                    # Measures ESI agreement against clinician-labeled vignettes
│   ├── vignettes.json          # Draft vignette set — see evals/README.md before trusting any number
│   ├── vignettes.py            # Pydantic schema + loader (pure — no LLM)
│   ├── metrics.py               # Exact-match / within-one / under-triage rate (pure — no LLM)
│   └── runner.py                 # Runs the real pipeline against reviewed vignettes — needs a real key
│
├── tests/
│   ├── test_assessment.py, test_retrieval.py, test_store.py, test_schema_utils.py,
│   │   test_eval_metrics.py, test_eval_vignettes.py
│   │                          # Pure logic — no API key, no network, no backend
│   ├── test_eval_runner.py    # Stubs the LLM call — needs the full pipeline stack to import, not a key
│   └── backend/               # API + DB tests — SQLite in-memory, still no LLM key needed
│
└── .github/workflows/
    └── tests.yml              # CI: pure-logic job (narrow deps) + full-suite job (everything)
```

---

## Setup

Two processes: the backend (FastAPI) and the UI (Streamlit). Both read the same `.env`.

### 1. Clone and install

```bash
git clone https://github.com/JustinOlivo52/triage-bot.git
cd triage-bot
pip install -r requirements.txt -r backend/requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in at minimum `ANTHROPIC_API_KEY` and `JWT_SECRET_KEY` (generate one: `python -c "import secrets; print(secrets.token_hex(32))"`). Everything else in `.env.example` has a documented default — see the comments there for what each one does.

### 3. Run the database migrations

```bash
python -m alembic -c backend/alembic.ini upgrade head
```

Creates a local SQLite file by default (`DATABASE_URL` in `.env.example`). Point `DATABASE_URL` at Postgres for anything beyond local dev.

### 4. Add clinical reference grounding (optional)

The committed `data/reference_index.json` already works out of the box (lexical search, no key needed). For semantic search, add `VOYAGE_API_KEY` to `.env` and rebuild:

```bash
python -m scripts.build_index
```

### 5. Launch both services

```bash
# Terminal 1
uvicorn backend.main:app --reload

# Terminal 2
streamlit run main.py
```

Set `DEMO_MODE=true` in `.env` before starting the backend to get a populated department and working demo logins on first run (see **Auth & Roles** above). Without it, the backend starts with an empty database and no accounts — create the first one directly, since account creation itself requires an existing admin:

```python
# one-off, from the repo root, with DATABASE_URL / JWT_SECRET_KEY set
from backend.db.session import SessionLocal
from backend.core.security import hash_password
from backend.models.user import User, UserRole

db = SessionLocal()
db.add(User(username="admin", hashed_password=hash_password("change-me"), full_name="Admin", role=UserRole.ADMIN))
db.commit()
```

---

## Tests

```bash
pip install -r requirements-dev.txt -r backend/requirements.txt
pytest
```

206 tests, well under a second for the deterministic core and well under a minute total, with **no API key and no network** — every LLM call any test would otherwise need is stubbed at the same boundary `agents/triage_agent.py` exposes for it.

Two CI jobs mirror that same split (`.github/workflows/tests.yml`):
- **pure-logic** — `agents/assessment.py`, `memory/`, and `rag/`'s lexical path, installed with a deliberately narrow dependency list (no LangChain, no FastAPI). If these tests ever start needing more than that, one of the "pure" modules has leaked a dependency it shouldn't have, and the job fails on purpose.
- **full-suite** — everything: the LangGraph pipeline, RAG's semantic-path imports, and the FastAPI/SQLAlchemy backend together.

Coverage includes: threshold tiering and boundary conditions, the extraction-to-findings mapping (including out-of-vocabulary rejection and the confirmed false positives that motivated the rewrite), age amplification, escalation derivation, FHIR model round-trips, JWT issue/verify and role enforcement per route, the full check-in→triage→persistence path with a stubbed LLM, and the audit trail (every mutating action produces exactly one row with the right actor/action/resource).

---

## Key Features

- **Every patient gets a score** — escalation is reported alongside the ESI level, never instead of it
- **Negation-aware symptom handling** — "denies chest pain" and "history of stroke in 2019" are read as what they are, via structured extraction on the existing triage call
- **Testable clinical rules** — all threshold and escalation logic lives in `agents/assessment.py` as pure functions with no LLM or network dependency
- **Single source of truth for thresholds** — the triage prompt renders its vital-sign ranges from `config.py`, so the prompt cannot drift from the code
- **Real auth, real roles** — JWT + bcrypt, three roles enforced server-side via one dependency, not scattered checks
- **FHIR-shaped persistence** — `Patient`/`Encounter`/`Observation`/`RiskAssessment` in a real database, with every simplification against true FHIR stated in the code
- **Append-only audit trail** — every mutating action logged, readable only by admin, survives even the end-of-shift reset
- **Structured LLM output** — Pydantic v2 schema-enforced responses via `with_structured_output(..., method="json_schema")`, not the default forced-tool-call method (which doesn't actually guarantee every field is populated — caught running against a real key)
- **Cost-aware escalation** — only `IMMEDIATE` patients trigger a physician-summary LLM call
- **Honest failure modes** — a pipeline failure is surfaced as a system error, never disguised as a clinical alert or a fabricated ESI score
- **Retrieval degrades in steps, not all-or-nothing** — semantic search (Voyage key) → lexical search (no key) → ungrounded reasoning (no index), never a crash
- **Zero-dependency clinical-rules testing** — 206 tests, no API key or network, split across two CI jobs so the pure logic's dependency guarantee is actually enforced, not just claimed
- **Eval harness, ready to run** — `evals/` measures ESI agreement against clinician-labeled vignettes (exact-match, within-one, under/over-triage rate); the harness itself is tested, the vignette set is drafted and awaiting clinical review before any number from it is a real claim

---

## V2 Roadmap

**Done (V1 — the clinical pipeline and its hardening)**
- [x] Separate ESI scoring from escalation so every patient is scored
- [x] Structured symptom extraction handling negation and history
- [x] Two-tier finding severity (critical vs concerning), single source of truth for thresholds
- [x] System failures distinguished from clinical alerts
- [x] Atomic writes, session-scoped storage for the (now superseded) file-backed store
- [x] RAG stack slimmed from ~1.6GB (torch/chromadb) to precomputed embeddings + numpy
- [x] Test suite over the deterministic clinical rules, running in CI

**Done (V2 — the FastAPI backend, FHIR model, auth, and full UI migration)**
- [x] Phase 1 — Backend skeleton: FastAPI, SQLAlchemy + Alembic, FHIR-shaped schema, JWT auth
- [x] Phase 2 — Vertical slice: check-in → triage → FHIR persistence, one real endpoint
- [x] Phase 3 — Append-only audit log wired to every mutating route, admin-only read endpoint
- [x] Phase 4 — Streamlit fully migrated off direct `agents/`/`memory/` imports onto the API; real login replaces the old anonymous demo-session model; nurse-override/disposition workflow added
- [x] CI fixed — a pre-existing dependency gap had left it silently red since the V1 RAG rewrite; now split into a narrow pure-logic job and a full-suite job, both verified in clean environments

**Done (eval harness — infrastructure, not yet a certified number)**
- [x] `evals/` built and tested: vignette schema, agreement/under-triage-rate metrics, a runner against the real pipeline — see `evals/README.md`
- [x] 20 starter vignettes spanning ESI 1-5, drafted against ESI v4 criteria

**Done (age-banded vital thresholds — pediatric drafted, not yet a certified number)**
- [x] Pediatric (age ≤5) gets its own `CRITICAL_VITALS_PEDIATRIC`/`CONCERNING_VITALS_PEDIATRIC` tables in `config.py`, applied throughout the pipeline (assessment, escalation, and the LLM prompt itself via `patient.age_group`)
- [x] Fixes the exact documented bug: a well 3-year-old at HR 110 / RR 26 no longer registers as tachycardic/tachypneic
- [x] Geriatric intentionally keeps adult thresholds — the real concern there (blunted response can mask severity) is handled by `assess_age_risk()` amplifying whatever finding *is* present, not by moving the numbers

**Done (returning patients consolidated onto one FHIR `Patient`)**
- [x] A returning patient (matched by name) reuses their existing `Patient` row and `patient_identifier` instead of getting a new chart number every visit — one person, one FHIR `Patient`, multiple `Encounter`s
- [x] `Patient.birth_date` is set once, at the first visit, and never overwritten; `Encounter.age_at_encounter` carries what was reported at each specific visit
- [x] `GET /encounters/{id}/prior-visits` now queries the FK directly instead of a name-matching join — simpler and actually correct rather than a heuristic

**Next**
- [ ] Clinical review of `evals/vignettes.json` (Justin), then a real run against `claude-opus-5` — would turn the model split in `config.py` from a reasoned default into a measured one
- [ ] Grow the vignette set toward the original 50-100 target once the starter batch is reviewed
- [ ] Clinical review of the pediatric vital thresholds in `config.py`/`data/esi_reference.md` — same gate as the eval vignettes, see the note in the reference doc
- [ ] Real identity resolution for returning patients (currently name-only matching — see Known Limitations)
- [ ] Database-level append-only enforcement on `audit_logs` (currently application-layer only)
- [ ] Deploy — needs a hosting decision for two services + a database, not just Streamlit Community Cloud (see `DEPLOY.md`)

---

## Known Limitations

- **Pediatric vital thresholds are drafted, not clinically reviewed yet.** They fix the previous adult-only bug (see below) but the specific numbers in `config.py`'s `CRITICAL_VITALS_PEDIATRIC`/`CONCERNING_VITALS_PEDIATRIC` need Justin's sign-off, same as `data/esi_reference.md`'s pediatric section and `evals/vignettes.json`.
- **No accuracy measurement yet.** There is no eval set, so the system's agreement with expert ESI assignment is currently unknown. Model choice per role is a reasoned default, not a measured one.
- **The fallback symptom scan is naive.** If the triage call fails, symptom detection degrades to a substring scan, which cannot handle negation — deliberate (a few false positives beat losing detection entirely on an already-degraded path), but findings on a system-error card should be read with that in mind.
- **`patient_identifier` generation isn't concurrency-safe.** It's a count-and-increment, not a DB sequence — fine for demo traffic, a real race under concurrent check-ins, and only applies the first time a given name is seen (a returning patient reuses their existing identifier, no new number generated).
- **Returning-patient matching is name-only.** Two different people who happen to share a name would incorrectly consolidate onto one `Patient` record; the same person spelled two different ways would incorrectly get two. Real identity resolution (DOB + name, or a patient-supplied identifier) would fix both, and isn't done here.
- **FHIR-shaped, not FHIR-certified.** No terminology binding, no `$validate`, no complete resource set — see the Data Model section above for exactly what's simplified and why.

---

## Background

Built by a current ED nurse transitioning into AI/health-tech engineering. The red flag criteria, vital thresholds, and ESI scoring logic reflect real-world ED triage practice and ESI v4 guidelines; the service architecture, auth model, and FHIR-shaped data model reflect what a health-tech engineering team would actually build around that clinical logic, not just a demo wrapped around a prompt.
