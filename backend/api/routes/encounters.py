"""
backend/api/routes/encounters.py — Check-in, triage, and the rest of the
encounter lifecycle: reading the queue, searching, prior visits, disposition,
and the admin reset. This is what main.py talks to instead of importing
agents/ and memory/ directly (V2_PLAN.md Phase 4).

Role split: any authenticated staff member can check patients in and read
the queue — that's ordinary clinical work, not a privileged action. Resolving
an encounter (a disposition decision) is physician+; the end-of-shift reset
is admin-only, matching V2_PLAN.md's auth table.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.api.deps import require_role
from backend.db.session import get_db
from backend.models.user import User, UserRole
from backend.schemas.encounter import CheckInRequest, EncounterOut, ResolveRequest
from backend.services.triage_service import (
    check_in_and_triage,
    get_encounter_by_patient_identifier,
    get_prior_visits,
    list_encounters,
    reset_all,
    resolve_encounter,
    search_encounters,
)

router = APIRouter(prefix="/encounters", tags=["encounters"])

_ANY_STAFF = require_role(UserRole.NURSE, UserRole.PHYSICIAN, UserRole.ADMIN)
_CAN_RESOLVE = require_role(UserRole.PHYSICIAN, UserRole.ADMIN)
_ADMIN_ONLY = require_role(UserRole.ADMIN)

_NOT_FOUND = HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Encounter not found")


@router.post("/check-in", response_model=EncounterOut, status_code=status.HTTP_201_CREATED)
def check_in(
    payload: CheckInRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(_ANY_STAFF),
) -> EncounterOut:
    result = check_in_and_triage(db, payload, current_user)
    return EncounterOut(encounter_id=result.encounter.id, card=result.card)


@router.get("", response_model=list[EncounterOut])
def list_all(
    active_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(_ANY_STAFF),
) -> list[EncounterOut]:
    results = list_encounters(db, active_only=active_only)
    return [EncounterOut(encounter_id=r.encounter.id, card=r.card) for r in results]


@router.get("/search", response_model=list[EncounterOut])
def search(
    q: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(_ANY_STAFF),
) -> list[EncounterOut]:
    results = search_encounters(db, q)
    return [EncounterOut(encounter_id=r.encounter.id, card=r.card) for r in results]


@router.get("/by-patient/{patient_identifier}", response_model=EncounterOut)
def get_by_patient_identifier(
    patient_identifier: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(_ANY_STAFF),
) -> EncounterOut:
    result = get_encounter_by_patient_identifier(db, patient_identifier)
    if result is None:
        raise _NOT_FOUND
    return EncounterOut(encounter_id=result.encounter.id, card=result.card)


@router.get("/{encounter_id}/prior-visits", response_model=list[EncounterOut])
def prior_visits(
    encounter_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(_ANY_STAFF),
) -> list[EncounterOut]:
    results = get_prior_visits(db, encounter_id)
    if results is None:
        raise _NOT_FOUND
    return [EncounterOut(encounter_id=r.encounter.id, card=r.card) for r in results]


@router.post("/{encounter_id}/resolve", response_model=EncounterOut)
def resolve(
    encounter_id: str,
    payload: ResolveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(_CAN_RESOLVE),
) -> EncounterOut:
    result = resolve_encounter(db, encounter_id, current_user, payload.note)
    if result is None:
        raise _NOT_FOUND
    return EncounterOut(encounter_id=result.encounter.id, card=result.card)


@router.post("/reset", status_code=status.HTTP_204_NO_CONTENT)
def reset(
    db: Session = Depends(get_db),
    current_user: User = Depends(_ADMIN_ONLY),
) -> None:
    reset_all(db, current_user)
