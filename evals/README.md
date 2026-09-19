# Eval Harness

Measures how often `agents/triage_agent.py`'s ESI score agrees with a
clinician's reference score on a set of synthetic vignettes. Turns the
`TRIAGE_MODEL`/`SUMMARY_MODEL` choice in `config.py` from a reasoned default
into a measured one.

## Status

The harness (`metrics.py`, `vignettes.py`, `runner.py`) is built and tested —
see `tests/test_eval_metrics.py`, `tests/test_eval_vignettes.py`,
`tests/test_eval_runner.py`, all zero-API-key. The 20 vignettes in
`vignettes.json` are **drafted, not reviewed**: every entry has
`reviewed_by_clinician: false`. They were written against documented ESI v4
decision points, not invented arbitrarily, but they have not been checked by
Justin (an actual ED nurse) yet — same gate `data/esi_reference.md` and
`data/seed_cohort.json` went through before either was trusted.

**No accuracy number from this harness means anything until that review
happens.** `evals/runner.py` enforces this: it refuses to run in its default
(reviewed-only) mode while `reviewed_by_clinician` is false on every
vignette, and running with `--include-draft` prints an explicit warning in
both the console output and the saved report.

## Review checklist (for Justin)

For each vignette in `vignettes.json`:
1. Does the chief complaint + vitals combination look like a real
   presentation, not a constructed puzzle?
2. Is `reference_esi` actually what you'd assign on the real ESI v4 flow?
3. Is `reference_rationale` accurate as written?

Flip `reviewed_by_clinician` to `true` only for vignettes you've actually
checked — partial review is fine, the runner works with however many
reviewed vignettes exist at the time.

The starter set is 20 vignettes (4-5 per ESI level). The original plan calls
for 50-100 — add more the same way, or ask for a fresh batch once these are
reviewed and any gaps in coverage are clearer.

## Running it

```bash
# Needs a real ANTHROPIC_API_KEY — this calls the actual model, not a stub.
# Roughly $0.05/vignette at claude-opus-5 pricing.
python -m evals.runner
```

Writes a timestamped JSON report to `evals/results/` (gitignored — see
`.gitignore` for why) and prints a summary: exact-match rate, within-one-ESI
rate, under-triage rate (the dangerous direction — assigned less urgent than
the reference), over-triage rate, and system-error rate (the pipeline failed
to produce a score at all).

```bash
python -m evals.runner --include-draft   # preview against unreviewed drafts too
```
