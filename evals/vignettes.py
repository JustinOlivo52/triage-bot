"""
evals/vignettes.py — Load and validate the clinician-labeled vignette set.

Pure pydantic — no LLM or network import, same as models.py itself, so this
stays testable without a key.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from models import VitalSigns

VIGNETTES_PATH = Path(__file__).parent / "vignettes.json"


class Vignette(BaseModel):
    """
    One synthetic patient presentation with a clinician-assigned reference
    ESI. `reviewed_by_clinician` gates whether the runner treats it as real
    ground truth or a draft — same review-before-trusted pattern already
    used for data/esi_reference.md and data/seed_cohort.json elsewhere in
    this project. Nothing here should be treated as a real accuracy claim
    until that flag is true.
    """

    id: str
    age: int = Field(..., ge=0, le=130)
    weight_kg: float = Field(..., gt=0, le=500)
    chief_complaint: str = Field(..., min_length=1)
    vitals: VitalSigns
    reference_esi: int = Field(..., ge=1, le=5)
    reference_rationale: str = Field(..., min_length=1)
    reviewed_by_clinician: bool = False


def load_vignettes(path: Path = VIGNETTES_PATH) -> list[Vignette]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Vignette.model_validate(v) for v in payload["vignettes"]]
