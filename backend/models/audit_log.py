"""
backend/models/audit_log.py — Append-only record of every mutating action.

Append-only is enforced at the application layer for V2: the audit router
(backend/api/routes/audit.py) exposes no update or delete endpoint, and
tests/backend/test_audit.py asserts none exists. True database-level
enforcement (a Postgres rule blocking UPDATE/DELETE on this table) is a
documented follow-up in V2_PLAN.md, not done here — SQLite, used in tests,
has no equivalent mechanism, so enforcing it only in Postgres would mean the
test suite could not verify the thing it's supposed to guarantee.
"""

import enum
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class AuditAction(str, enum.Enum):
    PATIENT_CHECK_IN = "patient_check_in"
    TRIAGE_RUN = "triage_run"
    TRIAGE_SYSTEM_ERROR = "triage_system_error"
    ESCALATION_RESOLVED = "escalation_resolved"
    PATIENT_VIEWED = "patient_viewed"
    USER_CREATED = "user_created"
    LOGIN = "login"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    actor_user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction), nullable=False, index=True)

    # Generic resource reference rather than a foreign key to one specific
    # table — a single audit log has to point at patients, encounters, and
    # users interchangeably.
    resource_type: Mapped[str] = mapped_column(String(30), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    metadata_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )

    @property
    def metadata_dict(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    @metadata_dict.setter
    def metadata_dict(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value, default=str)
