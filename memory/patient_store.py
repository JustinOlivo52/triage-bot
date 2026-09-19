"""
memory/patient_store.py — Patient queue with unique ID tracking.

Storage is pluggable (see memory/backends.py): a durable file for local use, or
per-session memory for the public demo. The store itself does not care which.

Storage format:
    {
        "id_counter": 5,
        "cards": {
            "PT-0001": { ...PatientCard... },
            "PT-0002": { ...PatientCard... }
        }
    }
"""

import json
import logging
from pathlib import Path
from typing import Iterable, MutableMapping

from config import BASE_DIR, PATIENT_ID_PREFIX, SEED_COHORT
from memory.backends import DictBackend, FileBackend, StorageBackend, empty_store
from models import Patient, PatientCard, TriageStatus

logger = logging.getLogger(__name__)

STORE_PATH: Path = BASE_DIR / "memory" / "patient_store.json"


class PatientStore:
    """
    Patient queue over a storage backend.

    Reads are cached for the life of the instance and invalidated on write. The
    previous version hit the backend on every method call, which meant a single
    Streamlit render performed six full reads and five complete
    deserialisations of every card in the queue.
    """

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        self._cache: dict | None = None

    # ── Internal I/O ─────────────────────────────────────────────────────────

    def _read(self) -> dict:
        if self._cache is None:
            self._cache = self._backend.read()
        return self._cache

    def _write(self, data: dict) -> None:
        self._backend.write(data)
        self._cache = data

    def refresh(self) -> None:
        """Drop the cached read. Call if the backing store changed elsewhere."""
        self._cache = None

    # ── ID Generation ─────────────────────────────────────────────────────────

    def _next_id(self, data: dict) -> str:
        """
        Increment the counter and return the next formatted patient ID.
        The caller persists the counter as part of its own write.
        """
        data["id_counter"] = data.get("id_counter", 0) + 1
        return f"{PATIENT_ID_PREFIX}-{data['id_counter']:04d}"

    # ── Returning Patient Detection ───────────────────────────────────────────

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Lowercase and strip for consistent name matching."""
        return name.strip().lower()

    def is_returning(self, name: str) -> bool:
        """Return True if any prior card exists for this patient name."""
        normalized = self._normalize_name(name)
        return any(
            self._normalize_name(card["patient"]["name"]) == normalized
            for card in self._read()["cards"].values()
        )

    def get_prior_visits(self, name: str) -> list[PatientCard]:
        """All prior cards for this name, oldest first."""
        normalized = self._normalize_name(name)
        matches = [
            card for card in self.get_all_cards()
            if self._normalize_name(card.patient.name) == normalized
        ]
        return sorted(matches, key=lambda c: c.patient.check_in_time)

    # ── Check-In ──────────────────────────────────────────────────────────────

    def check_in(self, patient: Patient) -> Patient:
        """
        Assign a unique ID, detect a returning patient, and persist a pending
        stub so the ID is reserved even if triage does not complete.
        """
        data = self._read()

        # Computed before the stub is written, or the patient would match
        # themselves.
        patient.is_returning = self.is_returning(patient.name)
        patient.patient_id = self._next_id(data)

        logger.info(
            "Check-in: %s assigned %s | returning=%s",
            patient.name, patient.patient_id, patient.is_returning,
        )

        stub = PatientCard(patient=patient)
        data["cards"][patient.patient_id] = json.loads(stub.model_dump_json())
        self._write(data)

        return patient

    # ── Save / Update ─────────────────────────────────────────────────────────

    def save_card(self, card: PatientCard) -> None:
        """Write or overwrite the card for a patient ID."""
        data = self._read()
        data["cards"][card.patient.patient_id] = json.loads(card.model_dump_json())
        self._write(data)
        logger.info("Saved card for %s — %s", card.patient.patient_id, card.display_esi)

    def seed(self, cards: Iterable[PatientCard]) -> None:
        """
        Replace the queue with a fixed set of cards.

        Used to load the demo cohort into a fresh session. The ID counter is
        advanced past the seeded IDs so visitor check-ins do not collide.
        """
        data = empty_store()
        highest = 0
        for card in cards:
            pid = card.patient.patient_id
            data["cards"][pid] = json.loads(card.model_dump_json())
            try:
                highest = max(highest, int(pid.rsplit("-", 1)[-1]))
            except ValueError:
                logger.warning("Seeded card has an unparseable ID: %s", pid)
        data["id_counter"] = highest
        self._write(data)
        logger.info("Seeded %d cards; next ID follows %d", len(data["cards"]), highest)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_card(self, patient_id: str) -> PatientCard | None:
        """Retrieve a single card by ID, or None if not found."""
        raw = self._read()["cards"].get(patient_id)
        if raw is None:
            logger.warning("Patient ID %s not found in store", patient_id)
            return None
        return PatientCard.model_validate(raw)

    def get_all_cards(self) -> list[PatientCard]:
        """All cards, newest check-in first."""
        cards = [PatientCard.model_validate(c) for c in self._read()["cards"].values()]
        return sorted(cards, key=lambda c: c.patient.check_in_time, reverse=True)

    def get_queue(self) -> list[PatientCard]:
        """
        The active queue: everyone not yet dispositioned.

        A patient leaves the queue when a clinician resolves them, not when the
        pipeline finishes scoring them.
        """
        return [
            c for c in self.get_all_cards()
            if c.patient.status is not TriageStatus.RESOLVED
        ]

    def search_by_name(self, name: str) -> list[PatientCard]:
        """Substring name search."""
        query = self._normalize_name(name)
        return [
            c for c in self.get_all_cards()
            if query in self._normalize_name(c.patient.name)
        ]

    def search_by_id(self, patient_id: str) -> PatientCard | None:
        """Direct ID lookup — the primary way staff locate a specific patient."""
        return self.get_card(patient_id.strip().upper())

    # ── Stats ─────────────────────────────────────────────────────────────────

    def queue_stats(self) -> dict:
        """Summary counts for the dashboard header."""
        all_cards = self.get_all_cards()
        active = [c for c in all_cards if c.patient.status is not TriageStatus.RESOLVED]
        return {
            "total": len(all_cards),
            "active": len(active),
            "immediate": sum(1 for c in active if c.needs_immediate_attention),
            "elevated": sum(
                1 for c in active
                if c.is_escalated and not c.needs_immediate_attention
            ),
            "routine": sum(1 for c in active if not c.is_escalated and c.triage_result),
            "pending": sum(1 for c in active if c.patient.status is TriageStatus.PENDING),
            "system_errors": sum(1 for c in all_cards if c.has_system_error),
        }

    # ── Admin ─────────────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Wipe the queue and reset the ID counter. Destructive."""
        self._write(empty_store())
        logger.warning("Patient store cleared — all records deleted")


# ─── Construction ─────────────────────────────────────────────────────────────

def file_store(path: Path = STORE_PATH) -> PatientStore:
    """Durable store shared by everyone using this instance of the app."""
    return PatientStore(FileBackend(path))


def session_store(container: MutableMapping) -> PatientStore:
    """
    Store scoped to one browser session.

    Used for the public demo so visitors cannot see, or clear, each other's
    patients. Nothing touches disk.
    """
    return PatientStore(DictBackend(container))


def load_seed_cards(path: Path = SEED_COHORT) -> list[PatientCard]:
    """
    Load the committed demo cohort.

    Returns an empty list if the file is absent or unreadable — a missing seed
    means an empty department, not a broken app.
    """
    if not path.exists():
        logger.info("No seed cohort at %s — starting with an empty queue", path)
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [PatientCard.model_validate(c) for c in payload["cards"].values()]
    except Exception as e:
        logger.error("Could not read seed cohort %s: %s", path, e)
        return []
