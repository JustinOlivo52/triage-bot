"""
memory/backends.py — Storage backends for the patient queue.

Two backends, because a locally-run department tool and a public demo have
opposite requirements:

    FileBackend  — durable, shared, atomic. One nurse station, one queue that
                   survives restarts.
    DictBackend  — in-memory over a caller-supplied mapping. The public demo
                   passes Streamlit's per-session state, so visitors get an
                   isolated queue and cannot see or clear each other's work.

DictBackend deliberately takes a plain MutableMapping rather than importing
streamlit, so this module stays free of UI dependencies and testable without
a browser session.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import MutableMapping, Protocol

logger = logging.getLogger(__name__)


def empty_store() -> dict:
    """
    A fresh, empty store.

    A function rather than a module constant on purpose: the previous version
    kept a single `_EMPTY_STORE` dict and handed out `.copy()` of it, which is
    a shallow copy — every "fresh" store aliased the same inner `cards` dict,
    so a reset followed by a check-in wrote the new patient into the constant.
    """
    return {"id_counter": 0, "cards": {}}


class StorageBackend(Protocol):
    """Where a PatientStore keeps its data."""

    def read(self) -> dict: ...
    def write(self, data: dict) -> None: ...


# ─── File Backend ─────────────────────────────────────────────────────────────

class FileBackend:
    """
    JSON file on disk, written atomically.

    Durable and shared across sessions, which is correct for a single
    department running the app locally.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict:
        if not self.path.exists():
            return empty_store()

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # Never silently discard a queue. The previous behaviour returned
            # an empty store on any read error, which turned one corrupt byte
            # into total, unannounced data loss. Preserve the file so it can be
            # recovered, and say so loudly.
            quarantine = self._quarantine(e)
            logger.error(
                "Patient store at %s is unreadable (%s). Moved to %s and "
                "started an empty queue. The original data is NOT lost.",
                self.path, e, quarantine,
            )
            return empty_store()

        if not isinstance(data, dict) or "cards" not in data:
            quarantine = self._quarantine("unexpected structure")
            logger.error(
                "Patient store at %s has an unexpected structure. Moved to %s "
                "and started an empty queue.",
                self.path, quarantine,
            )
            return empty_store()

        return data

    def _quarantine(self, reason: object) -> Path:
        """Move an unreadable store aside so it can be recovered by hand."""
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.path.with_suffix(f".corrupt-{stamp}.json")
        try:
            os.replace(self.path, target)
        except OSError as e:
            logger.error("Could not quarantine %s (%s): %s", self.path, reason, e)
        return target

    def write(self, data: dict) -> None:
        """
        Write atomically: full write to a temp file in the same directory,
        flush to disk, then rename over the target.

        os.replace is atomic on POSIX and Windows, so a crash mid-write leaves
        either the old file or the new one, never a half-written queue.
        """
        directory = self.path.parent
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".store-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.path)
        except Exception:
            # Leave the existing file untouched on any failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


# ─── Dict Backend ─────────────────────────────────────────────────────────────

class DictBackend:
    """
    In-memory store over any mutable mapping.

    The public demo passes Streamlit's session state, which scopes the queue to
    one browser session. Nothing is written to disk, which also suits hosts
    with ephemeral filesystems.
    """

    def __init__(self, container: MutableMapping, key: str = "_patient_store_data") -> None:
        self.container = container
        self.key = key

    def read(self) -> dict:
        data = self.container.get(self.key)
        if not isinstance(data, dict) or "cards" not in data:
            return empty_store()
        return data

    def write(self, data: dict) -> None:
        self.container[self.key] = data
