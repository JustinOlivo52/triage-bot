"""
memory/patient_store.py — Persistent patient queue with unique ID tracking.

Stores all PatientCard records to a JSON file on disk so the queue survives
app restarts and supports returning patient detection across sessions.

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

from config import BASE_DIR, PATIENT_ID_PREFIX
from models import Patient, PatientCard, TriageStatus

logger = logging.getLogger(__name__)

STORE_PATH: Path = BASE_DIR / "memory" / "patient_store.json"

_EMPTY_STORE: dict = {"id_counter": 0, "cards": {}}


# ─── PatientStore Class ───────────────────────────────────────────────────────

class PatientStore:
    """
    File-backed patient queue. Reads and writes to JSON on every operation
    so state is never lost between Streamlit reruns or app restarts.
    """

    def __init__(self, store_path: Path = STORE_PATH) -> None:
        self.store_path = store_path
        self._ensure_store()

    # ── Internal I/O ─────────────────────────────────────────────────────────

    def _ensure_store(self) -> None:
        """Create the store file if it does not exist."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write(_EMPTY_STORE.copy())
            logger.info("Patient store initialised at %s", self.store_path)

    def _read(self) -> dict:
        """Load the full store dict from disk."""
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read patient store: %s — resetting", e)
            return _EMPTY_STORE.copy()

    def _write(self, data: dict) -> None:
        """Write the full store dict to disk."""
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ── ID Generation ─────────────────────────────────────────────────────────

    def _next_id(self, data: dict) -> str:
        """
        Increment the counter and return the next formatted patient ID.
        Counter is written back to disk as part of the calling operation.
        """
        data["id_counter"] += 1
        return f"{PATIENT_ID_PREFIX}-{data['id_counter']:04d}"

    # ── Returning Patient Detection ───────────────────────────────────────────

    def _normalize_name(self, name: str) -> str:
        """Lowercase and strip for consistent name matching."""
        return name.strip().lower()

    def is_returning(self, name: str) -> bool:
        """Return True if any prior card exists for this patient name."""
        normalized = self._normalize_name(name)
        data = self._read()
        return any(
            self._normalize_name(card["patient"]["name"]) == normalized
            for card in data["cards"].values()
        )

    def get_prior_visits(self, name: str) -> list[PatientCard]:
        """
        Return all prior PatientCards for this name, sorted oldest first.
        Used to display visit history for returning patients.
        """
        normalized = self._normalize_name(name)
        data = self._read()
        matches = [
            PatientCard.model_validate(card)
            for card in data["cards"].values()
            if self._normalize_name(card["patient"]["name"]) == normalized
        ]
        return sorted(matches, key=lambda c: c.patient.check_in_time)

    # ── Check-In ──────────────────────────────────────────────────────────────

    def check_in(self, patient: Patient) -> Patient:
        """
        Assign a unique ID to the patient, detect if returning, and persist
        the bare patient record before triage begins.

        Returns the patient with patient_id and is_returning set.
        """
        data = self._read()

        patient_id = self._next_id(data)
        patient.patient_id = patient_id
        patient.is_returning = self.is_returning(patient.name)

        logger.info(
            "Check-in: %s assigned %s | returning=%s",
            patient.name, patient_id, patient.is_returning,
        )

        # Persist a pending stub so the ID is reserved even before triage completes
        stub_card = PatientCard(patient=patient)
        data["cards"][patient_id] = json.loads(stub_card.model_dump_json())
        self._write(data)

        return patient

    # ── Save / Update ─────────────────────────────────────────────────────────

    def save_card(self, card: PatientCard) -> None:
        """
        Write or overwrite the PatientCard for a given patient_id.
        Called after triage completes to replace the pending stub.
        """
        data = self._read()
        data["cards"][card.patient.patient_id] = json.loads(card.model_dump_json())
        self._write(data)
        logger.info("Saved card for %s — %s", card.patient.patient_id, card.display_esi)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_card(self, patient_id: str) -> PatientCard | None:
        """Retrieve a single PatientCard by ID. Returns None if not found."""
        data = self._read()
        raw = data["cards"].get(patient_id)
        if raw is None:
            logger.warning("Patient ID %s not found in store", patient_id)
            return None
        return PatientCard.model_validate(raw)

    def get_all_cards(self) -> list[PatientCard]:
        """
        Return all PatientCards sorted by check-in time, newest first.
        Used to populate the patient queue in the UI.
        """
        data = self._read()
        cards = [PatientCard.model_validate(c) for c in data["cards"].values()]
        return sorted(cards, key=lambda c: c.patient.check_in_time, reverse=True)

    def get_queue(self) -> list[PatientCard]:
        """
        Return the active triage queue: everyone not yet dispositioned.

        A patient leaves the queue when a clinician resolves them, not when the
        pipeline finishes scoring them.
        """
        return [
            c for c in self.get_all_cards()
            if c.patient.status is not TriageStatus.RESOLVED
        ]

    def search_by_name(self, name: str) -> list[PatientCard]:
        """Fuzzy name search — returns any card where the name contains the query."""
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
        """Return a summary dict used by the Streamlit dashboard header."""
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
        """
        Wipe the entire store and reset the ID counter.
        Intended for end-of-shift resets — destructive, use with confirmation.
        """
        self._write(_EMPTY_STORE.copy())
        logger.warning("Patient store cleared — all records deleted")


# ─── Module-Level Singleton ───────────────────────────────────────────────────

# Single shared instance imported by the triage agent and Streamlit UI.
store = PatientStore()
