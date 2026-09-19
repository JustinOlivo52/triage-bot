"""
backend/models/audit_log.py — Append-only record of every mutating action.

Append-only is enforced at two layers now:

  - Application layer: the audit router (backend/api/routes/audit.py)
    exposes no update or delete endpoint, and tests/backend/test_audit.py
    asserts none exists.
  - Database layer: a BEFORE UPDATE / BEFORE DELETE trigger on this table
    that aborts the statement outright, so even a raw SQL UPDATE or a bug
    in some future code path can't silently rewrite history.

The original version of this file claimed SQLite "has no equivalent
mechanism" to Postgres rules/triggers and left DB-level enforcement as a
documented follow-up for that reason. That claim was wrong — SQLite has
triggers too, just different syntax — so both dialects get one here,
registered via SQLAlchemy DDL events on `after_create` so they fire whether
the table is built by Alembic (real deployments) or by
Base.metadata.create_all() (the test suite, see tests/backend/conftest.py).
Row-level, so they apply to a bulk DELETE/UPDATE the same as a single-row one.
"""

import enum
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DDL, DateTime, Enum, ForeignKey, String, Text, event
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
    QUEUE_RESET = "queue_reset"


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


# ─── Database-level append-only enforcement ────────────────────────────────

_SQLITE_NO_UPDATE = DDL("""
    CREATE TRIGGER audit_logs_no_update
    BEFORE UPDATE ON audit_logs
    BEGIN
        SELECT RAISE(ABORT, 'audit_logs is append-only: UPDATE is not allowed');
    END
""")

_SQLITE_NO_DELETE = DDL("""
    CREATE TRIGGER audit_logs_no_delete
    BEFORE DELETE ON audit_logs
    BEGIN
        SELECT RAISE(ABORT, 'audit_logs is append-only: DELETE is not allowed');
    END
""")

# Postgres triggers need a function to call; one function, two triggers.
_POSTGRES_FUNCTION = DDL("""
    CREATE OR REPLACE FUNCTION audit_logs_append_only() RETURNS trigger AS $$
    BEGIN
        RAISE EXCEPTION 'audit_logs is append-only: % is not allowed', TG_OP;
    END;
    $$ LANGUAGE plpgsql
""")

_POSTGRES_NO_UPDATE = DDL("""
    CREATE TRIGGER audit_logs_no_update
    BEFORE UPDATE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only()
""")

_POSTGRES_NO_DELETE = DDL("""
    CREATE TRIGGER audit_logs_no_delete
    BEFORE DELETE ON audit_logs
    FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only()
""")

event.listen(AuditLog.__table__, "after_create", _SQLITE_NO_UPDATE.execute_if(dialect="sqlite"))
event.listen(AuditLog.__table__, "after_create", _SQLITE_NO_DELETE.execute_if(dialect="sqlite"))
event.listen(AuditLog.__table__, "after_create", _POSTGRES_FUNCTION.execute_if(dialect="postgresql"))
event.listen(AuditLog.__table__, "after_create", _POSTGRES_NO_UPDATE.execute_if(dialect="postgresql"))
event.listen(AuditLog.__table__, "after_create", _POSTGRES_NO_DELETE.execute_if(dialect="postgresql"))
