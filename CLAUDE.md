# triage-bot — Project Context

ED triage assistant: Streamlit UI, LangGraph pipeline, Claude for clinical
reasoning. Portfolio project — positioned for AI-engineer roles, with clinical
correctness and PHI handling as the differentiator. Built with Justin, a
current ED nurse transitioning into tech.

**Read this file fully before making changes.** It captures decisions and
rationale that aren't obvious from the code alone — several were learned the
hard way (see "Bugs found running against a real key" below).

## Where things stand

All core work is done and tested. **The only remaining step is deployment**,
which needs Justin's hosting account — see `DEPLOY.md` for the exact path.
Branch: `claude/codebase-analysis-improvements-mpz9oo` (12 commits ahead of
the original `main`). Read `git log --oneline` for the full sequence; each
commit message explains *why*, not just what.

122 tests pass with **no API key and no network** (`pytest`). If a change
ever makes that untrue, something has leaked a dependency it shouldn't have.

## Architecture

```
assess_node → triage_node → escalate_node → build_card_node
```

Linear, not branching. **This is the core design decision of the whole
project.** The original code treated "red flag" and "ESI score" as mutually
exclusive — a flagged patient got no score at all. That's wrong: acuity and
urgency are one clinical judgment. Now every patient is always scored, and
escalation (`NONE` / `ELEVATED` / `IMMEDIATE`) is derived from that score plus
deterministic findings, shown *alongside* the score, never instead of it.

- `assess_node` — vital signs only, against `config.py` thresholds. Fully
  deterministic, no LLM. A measured number against a threshold is not a
  judgment call and should never be delegated to a model.
- `triage_node` — the LLM call. Assigns ESI 1–5 **and** extracts symptom
  status (present/denied/historical) from the chief complaint in the same
  call — no extra API cost, since the model already reads the complaint to
  score it.
- `escalate_node` — combines vital + symptom findings, applies age risk last
  (it depends on whether anything else already fired), derives escalation,
  generates a physician summary **only if IMMEDIATE** (cost control).
- `agents/assessment.py` holds all the pure logic — zero LLM/network
  dependency, which is what makes the 122 tests instant and key-free.

## Structured symptom extraction

Replaced keyword matching, which couldn't tell "chest pain" from "denies
chest pain" or "history of chest pain." Now the LLM reports each vocabulary
term (from `config.py`) as `present` / `denied` / `historical`; only
`present` becomes a clinical finding. Vocabulary is supplied by us, not
invented by the model, so severity mapping stays deterministic.

Verified against real generated output (`data/seed_cohort.json`, PT-0007):
"denies chest pain, medication refill" → `{"symptom": "chest pain", "status":
"denied"}` → ESI 5, no escalation. That's the regression case — if this ever
breaks, this exact patient is the fastest way to notice.

## Two bugs found running against a real key

Every fix before this was verified with stubbed LLM calls. Both of these only
surfaced the first time real API calls ran, which is itself worth remembering
for future work here: **stub-only testing did not catch either of these.**

1. **`temperature=0` — rejected outright.** Current-gen models (Opus 5,
   Sonnet 5) removed sampling params in favor of adaptive thinking. Fix:
   omit `temperature` entirely rather than pass 0.
2. **Forced tool calls don't guarantee schema compliance.**
   `with_structured_output()` defaults to `method="function_calling"`
   (forced tool choice), which does not guarantee every required field is
   populated — a complex patient came back missing `esi_score` entirely.
   Fixed by switching to `method="json_schema"`, Claude's server-enforced
   structured-output feature. Also added `agents/schema_utils.py` as a
   defensive belt-and-suspenders layer for a related quirk (list fields
   occasionally returned as JSON-encoded strings).

If you add a new structured-output call anywhere in this codebase, use
`method="json_schema"` from the start — don't reintroduce the default.

## Model choice

`config.py`: `TRIAGE_MODEL = claude-opus-5` (the actual clinical judgment —
under-triage is the dangerous failure mode), `SUMMARY_MODEL = claude-sonnet-5`
(narrative from facts already decided). This is a **reasoned default, not a
measured one** — no eval harness exists yet to say which model is actually
right for each role. Tier 1.2 in the original plan (not yet started) is
building that eval set.

## Storage: two backends, on purpose

`memory/backends.py`: `FileBackend` (atomic write, durable, for local/
self-hosted use) and `DictBackend` (in-memory over a caller-supplied mapping,
used for `DEMO_MODE`). This exists because the original code was a
module-level file-backed singleton — every visitor to a public deployment
would have shared one queue and any visitor could wipe it with the reset
button. `DEMO_MODE=true` scopes each browser session to its own queue via
Streamlit session state and caps live triage runs (`LIVE_TRIAGE_LIMIT`,
default 3).

## RAG: intentionally not a vector database

`rag/` used to carry `sentence-transformers` + `torch` + `chromadb` to search
roughly 40 paragraphs — about 1.6GB of dependencies for a job a numpy dot
product handles in milliseconds. Rewritten: `data/esi_reference.md` (the
clinical reference, self-authored and clinically reviewed — see below) is
chunked on markdown headings, embedded once at build time via Voyage AI, and
searched with cosine similarity over a precomputed JSON index
(`data/reference_index.json`). Footprint dropped to ~531MB, which is what
makes free-tier hosting (~1GB ceiling) actually work.

Retrieval degrades in two steps, not one: semantic (Voyage key present) →
lexical (token overlap, no key needed) → no context (index missing). The app
is fully functional with just `ANTHROPIC_API_KEY`; the Voyage key is an
upgrade, never a requirement. `scripts/build_index.py --lexical` builds a
working index with zero embeddings provider at all.

## Clinical reference — reviewed, not a draft

`data/esi_reference.md` was self-authored (not copied from the ESI
handbook — check licensing before ever adding a third-party PDF to `data/`)
and has been **reviewed by Justin for clinical accuracy**. Its vital
thresholds must match `config.py` (`CRITICAL_VITALS` / `CONCERNING_VITALS`)
exactly — if they ever diverge, the code is right and the doc is wrong; fix
the doc.

## Seed cohort — real output, not fabricated

`data/seed_cohort.json` — 8 patients, ESI 1 through 5, generated by
`scripts/build_seed.py` against a real key. **Never hand-write results into
this file.** A portfolio piece claiming "this is what the system produces"
has to mean it. If the vignettes change, regenerate by running the script
again with a real `ANTHROPIC_API_KEY` (~$0.30).

## Testing philosophy

`agents/assessment.py`, `memory/backends.py`, and `rag/retriever.py`'s
lexical path are all pure functions with zero LLM/network dependency —
that's deliberate, not incidental, and it's why 122 tests run in well under a
second. Every regression this project has hit is pinned as a test case,
including the exact false-positive complaints that motivated the symptom
extraction rewrite. New clinical logic should follow the same pattern: keep
the decision function pure and put the LLM call in a thin wrapper around it.

## Known gaps (not yet started)

- Eval harness (clinician-labeled vignettes, measured agreement rate) — would
  turn the model choice above from reasoned into measured.
- Nurse override / disposition workflow.
- Audit log of triage decisions.
- Age-banded vital thresholds (current ones are adult-only).

## For a fresh Claude Code session picking this up

1. `git log --oneline` — read every commit message in order. They tell the
   real story better than this file summarizes it.
2. `pytest` — should be 122 passed, instantly, no key needed. If not, stop
   and figure out why before doing anything else.
3. `DEPLOY.md` — current deploy status and exact next steps.
4. Don't re-litigate the linear-pipeline decision or the `json_schema`
   structured-output choice without re-reading why they exist above — both
   were arrived at after a wrong version shipped first.
