"""backend/schemas/audit.py — Response shape for the read-only audit trail."""

from datetime import datetime

from pydantic import BaseModel

from backend.models.audit_log import AuditAction


class AuditLogOut(BaseModel):
    id: str
    actor_user_id: str
    actor_username: str
    action: AuditAction
    resource_type: str
    resource_id: str
    metadata: dict
    timestamp: datetime
