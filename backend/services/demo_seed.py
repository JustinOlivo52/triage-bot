"""
backend/services/demo_seed.py — Populate a fresh database when DEMO_MODE is
on: fixed demo accounts and the seeded patient cohort, so a public
deployment shows a working, populated department immediately instead of an
empty login screen with nothing to explore.

Both steps are idempotent — safe to run on every process startup, not just
the first one.
"""

import logging

from sqlalchemy.orm import Session

from backend.core.config import DEMO_ACCOUNT_PASSWORD
from backend.core.security import hash_password
from backend.models.patient import Patient as PatientRow
from backend.models.user import User, UserRole
from backend.services.triage_service import persist_seeded_card
from memory.patient_store import load_seed_cards

logger = logging.getLogger(__name__)

_DEMO_ACCOUNTS = [
    ("demo_nurse", "Demo Nurse", UserRole.NURSE),
    ("demo_physician", "Demo Physician", UserRole.PHYSICIAN),
    ("demo_admin", "Demo Admin", UserRole.ADMIN),
]


def seed_demo_accounts(db: Session) -> User:
    """Create the fixed demo accounts if they don't already exist yet.
    Returns demo_admin — the actor attributed to seeded patients below."""
    accounts: dict[str, User] = {}
    for username, full_name, role in _DEMO_ACCOUNTS:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            user = User(
                username=username,
                hashed_password=hash_password(DEMO_ACCOUNT_PASSWORD),
                full_name=full_name,
                role=role,
            )
            db.add(user)
            db.flush()
            logger.info("Seeded demo account: %s (%s)", username, role.value)
        accounts[username] = user
    db.commit()
    return accounts["demo_admin"]


def seed_demo_cohort(db: Session, actor: User) -> None:
    """Load data/seed_cohort.json into the shared queue, once. Skipped the
    moment any patient row exists at all — a real check-in always takes
    precedence, and this must never run again after that point."""
    if db.query(PatientRow).first() is not None:
        return

    cards = load_seed_cards()
    for card in cards:
        persist_seeded_card(db, card, actor)
    db.commit()
    logger.info("Seeded %d demo patient(s) into the department queue", len(cards))


def run_demo_seed(db: Session) -> None:
    admin = seed_demo_accounts(db)
    seed_demo_cohort(db, admin)
