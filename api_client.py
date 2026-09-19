"""
api_client.py — Thin HTTP client the Streamlit UI uses to talk to backend/.

This is the whole point of V2 Phase 4: main.py no longer imports agents/ or
memory/ at all. Every action — login, check-in, reading the queue, search,
resolve, reset — goes through this module as a real HTTP call carrying a
JWT, the same way any other client of the API would.

Functions take the bearer token explicitly rather than reaching into
st.session_state themselves, so this module has no Streamlit dependency and
could be reused (or tested) outside it.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import requests

from models import PatientCard

BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")

_TIMEOUT_SECONDS = 30.0


class ApiError(Exception):
    """Raised for any non-2xx response or a connection failure. `status_code`
    is None for a connection failure (backend unreachable), not an HTTP error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class Session(NamedTuple):
    access_token: str
    username: str
    role: str


class EncounterView(NamedTuple):
    encounter_id: str
    card: PatientCard


def _headers(session: Session) -> dict:
    return {"Authorization": f"Bearer {session.access_token}"}


def _request(method: str, path: str, session: Session | None = None, **kwargs) -> requests.Response:
    try:
        response = requests.request(
            method, f"{BACKEND_API_URL}{path}",
            headers=_headers(session) if session else None,
            timeout=_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.exceptions.ConnectionError as e:
        raise ApiError(f"Could not reach the backend at {BACKEND_API_URL}: {e}") from e
    except requests.exceptions.Timeout as e:
        raise ApiError(f"Backend request to {path} timed out.") from e

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(str(detail), status_code=response.status_code)
    return response


def _to_view(payload: dict) -> EncounterView:
    return EncounterView(encounter_id=payload["encounter_id"], card=PatientCard.model_validate(payload["card"]))


# ─── Health ────────────────────────────────────────────────────────────────────

def retrieval_mode() -> str:
    """Which retrieval path the backend's pipeline will use. Falls back to
    'none' on any failure — a health check should never crash the UI."""
    try:
        return _request("GET", "/health").json().get("retrieval_mode", "none")
    except ApiError:
        return "none"


# ─── Auth ──────────────────────────────────────────────────────────────────────

def login(username: str, password: str) -> Session:
    """OAuth2 password grant — form-encoded, per FastAPI's OAuth2PasswordRequestForm."""
    response = _request("POST", "/auth/login", data={"username": username, "password": password})
    token = response.json()["access_token"]

    me = requests.get(
        f"{BACKEND_API_URL}/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=_TIMEOUT_SECONDS,
    )
    me.raise_for_status()
    role = me.json()["role"]

    return Session(access_token=token, username=username, role=role)


# ─── Encounters ────────────────────────────────────────────────────────────────

def check_in(session: Session, *, full_name: str, age: int, weight_kg: float,
             chief_complaint: str, vitals: dict) -> EncounterView:
    response = _request("POST", "/encounters/check-in", session, json={
        "full_name": full_name, "age": age, "weight_kg": weight_kg,
        "chief_complaint": chief_complaint, "vitals": vitals,
    })
    return _to_view(response.json())


def list_encounters(session: Session, active_only: bool = False) -> list[EncounterView]:
    response = _request("GET", "/encounters", session, params={"active_only": active_only})
    return [_to_view(item) for item in response.json()]


def search(session: Session, query: str) -> list[EncounterView]:
    response = _request("GET", "/encounters/search", session, params={"q": query})
    return [_to_view(item) for item in response.json()]


def get_by_patient_identifier(session: Session, patient_identifier: str) -> EncounterView | None:
    try:
        response = _request("GET", f"/encounters/by-patient/{patient_identifier}", session)
    except ApiError as e:
        if e.status_code == 404:
            return None
        raise
    return _to_view(response.json())


def prior_visits(session: Session, encounter_id: str) -> list[EncounterView]:
    response = _request("GET", f"/encounters/{encounter_id}/prior-visits", session)
    return [_to_view(item) for item in response.json()]


def resolve(session: Session, encounter_id: str, note: str | None = None) -> EncounterView:
    response = _request("POST", f"/encounters/{encounter_id}/resolve", session, json={"note": note})
    return _to_view(response.json())


def reset_queue(session: Session) -> None:
    _request("POST", "/encounters/reset", session)
