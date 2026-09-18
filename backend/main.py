"""
backend/main.py — FastAPI app entry point.

Run with: uvicorn backend.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import auth
from backend.core.config import CORS_ORIGINS

app = FastAPI(
    title="Triage Bot API",
    description=(
        "Backend service for the ED triage assistant. FHIR-shaped data "
        "model (Patient, Encounter, Observation, RiskAssessment) — see "
        "V2_PLAN.md for what that does and does not mean."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)


@app.get("/health")
def health() -> dict:
    """Unauthenticated liveness check — no DB round-trip, for load balancers."""
    return {"status": "ok"}
