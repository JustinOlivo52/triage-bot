"""
backend/main.py — FastAPI app entry point.

Run with: uvicorn backend.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import auth, encounters
from backend.core.config import CORS_ORIGINS, DEMO_MODE
from backend.db.session import SessionLocal
from backend.services.demo_seed import run_demo_seed
from rag.retriever import retrieval_mode

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEMO_MODE:
        db = SessionLocal()
        try:
            run_demo_seed(db)
        except Exception:
            logger.exception("Demo seed failed — starting with whatever the database already had")
        finally:
            db.close()
    yield


app = FastAPI(
    title="Triage Bot API",
    description=(
        "Backend service for the ED triage assistant. FHIR-shaped data "
        "model (Patient, Encounter, Observation, RiskAssessment) — see "
        "V2_PLAN.md for what that does and does not mean."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(encounters.router)


@app.get("/health")
def health() -> dict:
    """
    Unauthenticated liveness check — no DB round-trip, for load balancers.

    Also reports which retrieval mode the pipeline will use (semantic /
    lexical / none) — main.py used to check this itself by importing rag/
    directly; now that Streamlit is a thin API client (V2_PLAN.md Phase 4),
    it asks the backend instead of reaching into the pipeline's internals.
    """
    try:
        mode = retrieval_mode()
    except Exception:
        logger.warning("Reference index unavailable", exc_info=True)
        mode = "none"
    return {"status": "ok", "retrieval_mode": mode}
