"""
backend/db/base.py — SQLAlchemy declarative base.

A separate module (rather than living in session.py) so Alembic's `env.py`
can import just the metadata without pulling in the engine/session machinery.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
