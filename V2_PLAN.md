# V2: From Demo to Company-Shaped Project

## Why

V1 proved the clinical logic works and is honestly tested. It's still shaped
like a demo: one Streamlit process holds the UI, the business logic, and the
only copy of the data, with no accounts and no record of who did what. V2's
goal is to make it *read* like a system a healthcare software team would
actually run internally — a real service boundary, real persistence, real
auth, and a data model that speaks the language an EHR-adjacent interviewer
already knows.

**Audience stays AI-engineer-primary.** That's why the frontend does not get
rebuilt (see decision below) — the effort goes where these interviews
actually probe: service design, data modeling, auth, and whether the AI
logic is production-shaped, not whether the UI is polished.

## Decision: FastAPI backend behind Streamlit, not a full frontend rewrite

Two options were on the table. Chose: keep Streamlit, but reduce it to a
thin client of a real FastAPI service — no more direct imports of `agents/`
or `memory/`, every action goes through an HTTP call with a JWT.

Rejected: full split with a new React/Next frontend. It's the more
impressive demo, but it spends build time on a skill (frontend engineering)
that isn't what these roles screen for, and it roughly doubles the surface
area to get right — real risk given time is unpredictable week to week. If
this ever gets asked about in an interview, "I scoped it to what the role
actually tests for" is a stronger answer than a half-finished frontend.

## Target shape

```
triage-bot/
├── main.py                  # Streamlit — becomes a thin API client
├── agents/                  # UNCHANGED — the triage pipeline itself is not being rewritten
├── models.py                # UNCHANGED — Pydantic domain models used inside the pipeline
├── config.py                # UNCHANGED
├── rag/                     # UNCHANGED
│
├── backend/                 # NEW — the real service
│   ├── main.py               # FastAPI app, router registration
│   ├── core/
│   │   ├── config.py          # backend-specific settings (DB URL, JWT secret, token TTL)
│   │   └── security.py        # password hashing, JWT issue/verify, role dependencies
│   ├── db/
│   │   ├── session.py         # engine + session factory (Postgres in prod, SQLite for tests)
│   │   └── base.py            # SQLAlchemy declarative base
│   ├── models/                # SQLAlchemy ORM models — FHIR-shaped (see below)
│   │   ├── user.py             # accounts + roles
│   │   ├── patient.py          # FHIR Patient
│   │   ├── encounter.py        # FHIR Encounter — the ED visit
│   │   ├── observation.py      # FHIR Observation — one row per vital sign
│   │   ├── risk_assessment.py  # FHIR RiskAssessment — the ESI score + escalation
│   │   └── audit_log.py        # append-only action log
│   ├── schemas/                # Pydantic request/response models for the API
│   ├── services/               # business logic — wraps agents/triage_agent.py, unchanged pipeline
│   ├── api/
│   │   ├── deps.py             # get_db, get_current_user, require_role(...)
│   │   └── routes/
│   │       ├── auth.py          # POST /auth/login, /auth/me
│   │       ├── patients.py      # check-in, list, get, search
│   │       ├── encounters.py    # triage run, escalation, resolve/disposition
│   │       └── audit.py         # read-only audit trail, role-gated
│   └── alembic/                 # migrations — the record of how the schema evolved
│
└── tests/
    ├── (existing 122 — untouched, still zero-dependency)
    └── backend/                 # new — API + DB tests, SQLite in-memory, still no LLM key needed
```

**The existing 122 tests do not change.** `agents/assessment.py` and the
rest of the pure-function core stay exactly as tested. The backend wraps
them; it doesn't replace them.

## Data model: FHIR-shaped, not FHIR-certified

Honest framing, stated once here so it doesn't drift into overclaiming
anywhere else: this maps our domain onto **real FHIR resource types and
their actual field names**, which is a legitimate signal to anyone who
knows EHR data (EPIC, Cerner, any HL7-conformant shop). It is **not** a
certified FHIR server — no full terminology binding, no `$validate`, no
complete resource set. That distinction goes in the README's V2 section
verbatim.

| Our concept | FHIR resource | Why this one |
|---|---|---|
| A person | `Patient` | identifier, name, birthDate, gender |
| One ED visit | `Encounter` | status, class=`EMER`, period, reasonCode (chief complaint) |
| One vital sign reading | `Observation` | code (LOINC), valueQuantity, effectiveDateTime — **one row per vital**, not one JSON blob, because that's how a real EHR stores it and it's what makes the audit trail meaningful |
| ESI score + escalation | `RiskAssessment` | subject, encounter, basis, prediction.outcome, prediction.qualitativeRisk, rationale — this is the actual FHIR resource for "here is an acuity/risk judgment about this patient," which is exactly what triage is |

`agents/triage_agent.py` still returns the existing `PatientCard` Pydantic
model — that's the pipeline's natural output shape and it stays that way.
The **service layer** (`backend/services/`) is what maps a `PatientCard`
into `Patient` + `Encounter` + `Observation` rows + a `RiskAssessment` row
at persistence time, and reassembles them back into a `PatientCard` for the
API response. The pipeline doesn't know FHIR exists; only the boundary does.

## Auth and roles

JWT bearer tokens (`python-jose`), passwords hashed with bcrypt
(`passlib`). Three roles: `nurse` (check in patients, run triage),
`physician` (everything a nurse can do, plus resolve/override escalations),
`admin` (both, plus read the audit log and manage accounts). Enforced via a
FastAPI dependency (`require_role(...)`), not scattered `if` checks in route
bodies — one place to audit for correctness.

## Audit log

Every mutating action — check-in, triage run, escalation resolution,
account creation — writes one `AuditLog` row: `actor_user_id`, `action`,
`resource_type`, `resource_id`, `timestamp`, `metadata` (JSON). Append-only
at the application layer: the audit router exposes no update or delete
endpoint, and a test asserts none exists. (A future hardening pass could add
a Postgres rule blocking `UPDATE`/`DELETE` at the database layer too — noted
as a follow-up, not required for V2.)

## Database

Postgres in the target deployment; SQLite in-memory for the test suite, so
the existing zero-dependency testing philosophy holds for the new tests too
— no Postgres instance required to run `pytest`. SQLAlchemy is the
abstraction that makes this free; Alembic migrations are still written
against Postgres syntax where the two diverge, and tests catch anything
that doesn't.

## Phases (each one is a working, tested, committed slice)

1. **Backend skeleton** — FastAPI app, DB session plumbing, SQLAlchemy
   models (User, Patient, Encounter, Observation, RiskAssessment,
   AuditLog), first Alembic migration, auth (login + JWT + role
   dependency). No triage logic wired yet. *Tests: models round-trip,
   auth issues/verifies tokens, role dependency rejects correctly.*
2. **Vertical slice** — one real endpoint: check-in → run the existing
   `agents/triage_agent.py` pipeline → persist as FHIR-shaped rows → return
   a `PatientCard`. Streamlit's check-in form calls this endpoint instead of
   importing `agents/` directly. *Tests: full round-trip through the API
   with a stubbed LLM, matching how `agents/` is already tested.*
3. **Audit log wired everywhere** — every mutating route writes its row;
   read endpoint gated to `admin`. *Tests: every mutating action produces
   exactly one audit row with the right actor/action/resource.*
4. **Full migration** — every remaining Streamlit action (search, queue
   view, resolve/disposition, reset) moves off direct `memory/` calls onto
   the API. `memory/backends.py` becomes backend-internal; Streamlit no
   longer imports it at all.
5. **README + architecture diagram** — document the service boundary, the
   FHIR mapping table above, and the auth model, so the project explains
   itself to a reader who never talks to us.

Given how variable available time is, phases are the unit of "done enough to
stop for now" — each one lands as its own commit(s) with passing tests, and
the next session (here or in VS Code) can pick up at the next phase using
this file the same way `CLAUDE.md` is used for the rest of the project.

## Explicitly out of scope for V2

- Rebuilding the frontend (see decision above).
- Full FHIR server compliance (`$validate`, complete terminology bindings).
- The eval harness (still the right next thing after V2, not part of it —
  see `CLAUDE.md` → Known gaps).
