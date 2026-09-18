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

from dotenv import load_dotenv

# Unlike the top-level config.py, this module previously read only real
# environment variables — a .env file worked for Streamlit (which does call
# load_dotenv()) but silently did nothing for the backend, so JWT_SECRET_KEY
# etc. would only ever come from a real export. Loading it here too means
# one .env file configures both processes, which is what the README's setup
# instructions assume.
load_dotenv()


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

# ─── Demo Mode ────────────────────────────────────────────────────────────────
#
# Deliberately its own flag rather than importing DEMO_MODE from the
# top-level config.py, for the same reason risk_assessment.py doesn't import
# EscalationLevel from models.py: backend/ has no import-time dependency on
# the pipeline side of the repo. Set the same env var name on both processes
# in a real deployment and they agree without either importing the other.
#
# On, this seeds fixed demo accounts and the demo patient cohort into the
# shared database at startup (idempotent — safe on every restart) so a
# public deployment shows a populated department immediately, the same way
# V1's DEMO_MODE auto-loaded the seed cohort into a session-scoped store.
DEMO_MODE: bool = _bool_env("DEMO_MODE")
DEMO_ACCOUNT_PASSWORD: str = os.getenv("DEMO_ACCOUNT_PASSWORD", "demo1234")
