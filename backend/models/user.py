"""backend/models/user.py — Accounts and roles."""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


class UserRole(str, enum.Enum):
    """
    Three roles, enforced by backend/api/deps.py::require_role, not by
    scattered checks in route bodies — one place to audit for correctness.
    """
    NURSE = "nurse"          # check in patients, run triage
    PHYSICIAN = "physician"  # + resolve/override escalations
    ADMIN = "admin"          # + read the audit log, manage accounts


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
