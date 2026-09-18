"""
backend/schemas/encounter.py — Request/response shapes for the check-in route.

`vitals` reuses the top-level `models.VitalSigns` directly rather than
redefining the same six fields with the same constraints here — it's already
exactly the shape intake collects. `CheckInResponse` reuses `models.PatientCard`
the same way, per V2_PLAN.md: the pipeline's native output shape is the API's
response shape, and only the service layer in between knows FHIR exists.
"""

from pydantic import BaseModel, Field

from models import PatientCard, VitalSigns


class CheckInRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=200)
    age: int = Field(..., ge=0, le=130)
    weight_kg: float = Field(..., gt=0, le=500)
    chief_complaint: str = Field(..., min_length=1)
    vitals: VitalSigns


class CheckInResponse(BaseModel):
    encounter_id: str
    card: PatientCard
