"""
backend/services/audit_service.py — Read-only access to the audit trail.

No write functions here on purpose: every write happens at the point of the
action it records (backend/api/routes/auth.py, triage_service.py, ...),
right alongside the change it's documenting. A generic "write an audit row"
helper here would invite audit calls to drift away from the code that
actually did the thing — one more place to forget to call it.
"""

from sqlalchemy.orm import Session

from backend.models.audit_log import AuditAction, AuditLog
from backend.models.user import User


def list_audit_logs(
    db: Session,
    *,
    action: AuditAction | None = None,
    resource_id: str | None = None,
    limit: int = 200,
) -> list[tuple[AuditLog, str]]:
    """Newest first. Returns (row, actor_username) pairs so a caller doesn't
    need a second round trip just to show who did what."""
    query = db.query(AuditLog, User.username).join(User, AuditLog.actor_user_id == User.id)
    if action is not None:
        query = query.filter(AuditLog.action == action)
    if resource_id is not None:
        query = query.filter(AuditLog.resource_id == resource_id)
    return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
