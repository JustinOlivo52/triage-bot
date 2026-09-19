"""
tests/backend/conftest.py — Shared fixtures for backend tests.

Every backend test runs against a fresh in-memory SQLite database, created
and torn down per test. No Postgres, no network, no API key — same
zero-dependency philosophy the rest of the project's 122 tests already hold
to, extended to cover the backend as well.
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret-do-not-use-in-production")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db.base import Base
from backend.db.session import get_db
from backend.main import app
from backend.models.user import User, UserRole
from backend.core.security import hash_password


@pytest.fixture()
def db_session():
    """
    A fresh in-memory SQLite database per test.

    StaticPool + check_same_thread=False keeps a single in-memory DB alive
    across the connections FastAPI's TestClient opens per request — without
    it, each connection would get its own empty :memory: database.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture()
def client(db_session):
    """FastAPI TestClient with get_db overridden to the test session."""
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def make_user(db_session, username: str, role: UserRole, password: str = "testpass123") -> User:
    """Create and persist a user directly, bypassing the API — for test setup."""
    user = User(
        username=username,
        hashed_password=hash_password(password),
        full_name=username.title(),
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def auth_headers(client, username: str, password: str = "testpass123") -> dict:
    """Log in and return an Authorization header for subsequent requests."""
    r = client.post("/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
