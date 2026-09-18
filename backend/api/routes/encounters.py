"""
backend/api/routes/encounters.py — Check-in + triage: the vertical slice.

Any authenticated staff member can check a patient in and run triage — this
isn't a privileged action the way resolving/overriding an escalation
(physician+) or reading the audit log (admin-only) is, so the role list
below is all three roles rather than a subset.
"""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.api.deps import require_role
from backend.db.session import get_db
from backend.models.user import User, UserRole
from backend.schemas.encounter import CheckInRequest, CheckInResponse
from backend.services.triage_service import check_in_and_triage

router = APIRouter(prefix="/encounters", tags=["encounters"])

_ANY_STAFF = require_role(UserRole.NURSE, UserRole.PHYSICIAN, UserRole.ADMIN)


@router.post("/check-in", response_model=CheckInResponse, status_code=status.HTTP_201_CREATED)
def check_in(
    payload: CheckInRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(_ANY_STAFF),
) -> CheckInResponse:
    result = check_in_and_triage(db, payload, current_user)
    return CheckInResponse(encounter_id=result.encounter.id, card=result.card)
