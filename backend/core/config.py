"""
backend/core/config.py — Backend service configuration.

Kept separate from the top-level config.py deliberately: that file configures
the triage *pipeline* (models, thresholds, retrieval) and is imported by both
the backend and the still-standalone Streamlit code paths during the V2
migration. This file configures the *service* (database, auth) and is only
ever imported inside backend/.
"""

import os
import secrets


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


# ─── Database ─────────────────────────────────────────────────────────────────
#
# Postgres in the real deployment. SQLite in-memory for tests — that keeps the
# project's existing testing philosophy intact: the test suite needs no
# external service, Postgres included.
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./triage_bot.db")


# ─── Auth ─────────────────────────────────────────────────────────────────────
#
# JWT_SECRET_KEY has no safe default in production: generating a random one
# per process would invalidate every issued token on restart, and a
# committed default would mean every deployment of this repo shares a signing
# key. Fail loudly instead, except under pytest, where a fixed key keeps
# tests deterministic without needing a real secret.
_RUNNING_UNDER_PYTEST = "PYTEST_CURRENT_TEST" in os.environ

JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "")
if not JWT_SECRET_KEY:
    if _RUNNING_UNDER_PYTEST:
        JWT_SECRET_KEY = "test-only-secret-not-for-production-" + secrets.token_hex(8)
    else:
        raise RuntimeError(
            "JWT_SECRET_KEY is not set. Generate one and set it as an "
            "environment variable — do not commit a default:\n\n"
            "  python -c \"import secrets; print(secrets.token_hex(32))\""
        )

JWT_ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # one shift

# ─── CORS ─────────────────────────────────────────────────────────────────────
#
# The Streamlit client and the API may run as separate processes/origins
# during local dev. Locked down by default; widen explicitly per deployment.
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()
]
