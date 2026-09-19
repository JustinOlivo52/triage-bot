"""
backend/schemas/encounter.py — Request/response shapes for encounter routes.

`vitals` reuses the top-level `models.VitalSigns` directly rather than
redefining the same six fields with the same constraints here — it's already
exactly the shape intake collects. `EncounterOut` reuses `models.PatientCard`
the same way, per V2_PLAN.md: the pipeline's native output shape is the API's
response shape, and only the service layer in between knows FHIR exists. It's
the response for every encounter route — check-in, get, list, search, resolve
all return the same {encounter_id, card} shape.
"""

from pydantic import BaseModel, Field

from models import PatientCard, VitalSigns


class CheckInRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=200)
    age: int = Field(..., ge=0, le=130)
    weight_kg: float = Field(..., gt=0, le=500)
    chief_complaint: str = Field(..., min_length=1)
    vitals: VitalSigns


class EncounterOut(BaseModel):
    encounter_id: str
    card: PatientCard


class ResolveRequest(BaseModel):
    note: str | None = Field(default=None, max_length=2000)
