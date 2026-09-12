"""
scripts/build_seed.py — Generate the seeded demo cohort.

Run once with a real API key, then commit the result:

    export ANTHROPIC_API_KEY=...
    python -m scripts.build_seed
    git add data/seed_cohort.json

Costs roughly $0.30 — a handful of real triage runs on the configured models.

**These are real model outputs on purpose.** Hand-authoring plausible-looking
triage results and committing them as if the system produced them would be
dishonest in a portfolio piece, and it falls apart the moment anyone asks how
a particular score was reached. If the cohort needs changing, change the
vignettes and re-run.

The vignettes are synthetic. No real patient data is used anywhere in this
repository.
"""

import json
import logging
import sys

from agents.triage_agent import run_triage
from config import ANTHROPIC_API_KEY, SEED_COHORT
from models import Patient, VitalSigns

log = logging.getLogger("build_seed")

NORMAL = dict(
    heart_rate=76, systolic_bp=124, diastolic_bp=78,
    respiratory_rate=16, spo2=98.0, temperature_c=36.8,
)

# A spread across the full acuity range, chosen so a visitor sees the system's
# whole behaviour: immediate escalation, elevated concern, routine scoring, and
# the negation handling that the keyword scan used to get wrong.
VIGNETTES: list[dict] = [
    dict(
        name="Maria Alvarez", age=78, weight_kg=61.0,
        chief_complaint="fever and new confusion since yesterday, not eating or drinking",
        vitals=dict(heart_rate=134, respiratory_rate=32, spo2=89.0, temperature_c=39.4,
                    systolic_bp=96, diastolic_bp=58),
    ),
    dict(
        name="Daniel Okafor", age=62, weight_kg=94.0,
        chief_complaint="crushing chest pain radiating to jaw and left arm, diaphoretic, started 40 minutes ago",
        vitals=dict(heart_rate=112, systolic_bp=88, diastolic_bp=54, spo2=93.0),
    ),
    dict(
        name="Grace Lindqvist", age=34, weight_kg=68.0,
        chief_complaint="sudden severe headache while lifting, worst of my life, photophobia",
        vitals=dict(heart_rate=98, systolic_bp=168, diastolic_bp=96),
    ),
    dict(
        name="Priya Raman", age=58, weight_kg=70.0,
        chief_complaint="prior episode of syncope last year, today has chest pressure with exertion",
        vitals=dict(systolic_bp=186, diastolic_bp=104, heart_rate=88),
    ),
    dict(
        name="Tom Becker", age=35, weight_kg=81.0,
        chief_complaint="mild abdominal pain for 3 days, eating and drinking normally, no vomiting",
        vitals={},
    ),
    dict(
        name="Sam Whitfield", age=44, weight_kg=77.0,
        chief_complaint="laceration to left forearm from kitchen knife, bleeding controlled with pressure",
        vitals=dict(heart_rate=92),
    ),
    dict(
        name="Nina Castellanos", age=29, weight_kg=64.0,
        chief_complaint="denies chest pain, here for a medication refill and blood pressure check",
        vitals={},
    ),
    dict(
        name="Alice Nguyen", age=25, weight_kg=59.0,
        chief_complaint="stubbed toe on bedframe this morning, sore but weight bearing",
        vitals={},
    ),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    if not ANTHROPIC_API_KEY:
        log.error(
            "ANTHROPIC_API_KEY is not set.\n\n"
            "  export ANTHROPIC_API_KEY=...\n"
            "  python -m scripts.build_seed\n\n"
            "This script runs real triage calls (roughly $0.30 total). It will "
            "not fabricate results."
        )
        return 1

    cards = []
    for i, vignette in enumerate(VIGNETTES, start=1):
        patient_id = f"PT-{i:04d}"
        patient = Patient(
            patient_id=patient_id,
            name=vignette["name"],
            age=vignette["age"],
            weight_kg=vignette["weight_kg"],
            chief_complaint=vignette["chief_complaint"],
            vitals=VitalSigns(**{**NORMAL, **vignette["vitals"]}),
        )

        log.info("Triaging %s — %s", patient_id, vignette["name"])
        card = run_triage(patient)

        if card.has_system_error:
            log.error(
                "%s failed: %s\nAborting — a seed cohort containing system "
                "errors would misrepresent the system.",
                patient_id, card.escalation.system_error,
            )
            return 1

        cards.append(card)
        log.info("  → %s / %s", card.display_esi, card.escalation.level.value)

    SEED_COHORT.parent.mkdir(parents=True, exist_ok=True)
    SEED_COHORT.write_text(
        json.dumps(
            {"cards": {c.patient.patient_id: json.loads(c.model_dump_json()) for c in cards}},
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info("Wrote %d cards to %s", len(cards), SEED_COHORT)
    log.info("Commit it: git add %s", SEED_COHORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
