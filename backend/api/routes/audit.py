"""
backend/api/routes/audit.py — Read-only audit trail, admin-only.

Completes V2_PLAN.md's Phase 3: every mutating route already writes its
row (that landed alongside each route as it was built, rather than being
retrofitted later); this is the one piece that was still missing — somewhere
an admin can actually read the trail back.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.api.deps import require_role
from backend.db.session import get_db
from backend.models.audit_log import AuditAction
from backend.models.user import User, UserRole
from backend.schemas.audit import AuditLogOut
from backend.services.audit_service import list_audit_logs

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditLogOut])
def read_audit_log(
    action: AuditAction | None = None,
    resource_id: str | None = None,
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.ADMIN)),
) -> list[AuditLogOut]:
    rows = list_audit_logs(db, action=action, resource_id=resource_id, limit=limit)
    return [
        AuditLogOut(
            id=row.id,
            actor_user_id=row.actor_user_id,
            actor_username=username,
            action=row.action,
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            metadata=row.metadata_dict,
            timestamp=row.timestamp,
        )
        for row, username in rows
    ]
