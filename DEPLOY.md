# Deploying Triage Bot

Three things only you can do — a key I don't have, and judgment I shouldn't
make for you. Everything else is already built and tested. This doc is the
exact path from here to a live link.

---

## Step 1 — Review the clinical reference (no key needed)

Read `data/esi_reference.md`. It's self-authored and cited, not copied from
the ESI handbook, and it's marked DRAFT until you sign off on it. Check in
particular:

- The vital sign thresholds match `config.py` (`CRITICAL_VITALS` /
  `CONCERNING_VITALS`) exactly. If they don't, the code is right and the doc
  is wrong.
- The clinical framing — decision points, high-risk presentations, age
  modifiers — reads true to how you actually triage.

When you're satisfied, delete the `> Status: DRAFT` line at the top. That's
the whole review gate.

## Step 2 — Generate the seed cohort (needs your Anthropic key)

The demo needs real triage output, not hand-written fake results — that's
the difference between an honest artifact and something that falls apart
the moment someone asks "how did it score this."

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install -r requirements.txt
python -m scripts.build_seed
```

Costs about $0.30 (8 real triage runs across Opus 5 and Sonnet 5). Writes
`data/seed_cohort.json`. Commit it:

```bash
git add data/seed_cohort.json
git commit -m "Add seeded demo cohort"
git push
```

Optional — better retrieval, needs a Voyage key:

```bash
export VOYAGE_API_KEY=...
python -m scripts.build_index
git add data/reference_index.json && git commit -m "Semantic reference index" && git push
```

Skip this if you don't have a Voyage key. The committed lexical index already
works — it's a weaker search, not a missing one.

## Step 3 — Deploy (needs your hosting account)

**Recommended: Streamlit Community Cloud.** Free, zero config file, connects
straight to your GitHub repo.

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub.
2. New app → pick `JustinOlivo52/triage-bot`, branch
   `claude/codebase-analysis-improvements-mpz9oo` (or `main` once you've
   merged), main file `main.py`.
3. Before deploying, open **Advanced settings → Secrets** and paste:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   DEMO_MODE = "true"
   LIVE_TRIAGE_LIMIT = "3"
   ```

   Add `VOYAGE_API_KEY = "..."` too if you built the semantic index in Step 2.

4. Deploy. First load takes 30-60s to install dependencies (531MB, well under
   the ~1GB free-tier ceiling — this was the whole point of the stack
   rewrite).

**`DEMO_MODE=true` is what makes this safe to share.** Without it, every
visitor shares one patient queue and any visitor can wipe it. With it, each
visitor gets their own session-scoped queue seeded from your committed
cohort, and live triage is capped at 3 runs per session.

Render or Fly work too if you'd rather not use Streamlit's own host — same
env vars, same footprint, just needs a `Procfile` or `render.yaml` I can
write on request.

## Step 4 — Verify

Open the URL in a private/incognito window (a clean session, no cookies):

- [ ] Seeded patients are visible immediately in the Active Queue
- [ ] Opening a card shows escalation + ESI together, and the Symptom
      Extraction panel for a denied/historical complaint
- [ ] Checking in a new patient runs a live triage and decrements the counter
      under the submit button
- [ ] After 3 live runs, the 4th shows the "Demo limit reached" message
- [ ] "Reset My Demo Session" only clears your own view
- [ ] No key is visible in page source or browser devtools network tab

Then put the URL at the top of the README and you're done.

---

## If you want me to keep going without a key

I can prepare Render/Fly config, tighten the README further, or start the
eval harness (Tier 1.2 — scoring the system against clinician-labeled
vignettes) while you handle Steps 2-3. Say the word.
