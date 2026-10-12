"""
tests/backend/test_rate_limit.py — backend/core/rate_limit.py.

Two layers: the RateLimiter class itself (unit, fake clock, no HTTP), and
the wired-up check-in endpoint (integration, via TestClient — the real
pipeline is stubbed the same way test_encounters.py does it).
"""

import backend.services.triage_service as triage_service
from backend.core.rate_limit import RateLimiter, check_in_limiter
from backend.models.user import UserRole
from models import EscalationAssessment, EscalationLevel, Patient as DomainPatient, PatientCard, TriageResult
from tests.backend.conftest import auth_headers, make_user

VALID_VITALS = {
    "heart_rate": 88, "systolic_bp": 120, "diastolic_bp": 78,
    "respiratory_rate": 16, "spo2": 98.0, "temperature_c": 37.0,
}


def _check_in_payload(name="Jane Doe"):
    return {
        "full_name": name,
        "age": 40,
        "weight_kg": 70.0,
        "chief_complaint": "denies chest pain, medication refill",
        "vitals": VALID_VITALS,
    }


def _stub_run_triage(monkeypatch) -> None:
    def _fake(domain_patient: DomainPatient) -> PatientCard:
        return PatientCard(
            patient=domain_patient,
            triage_result=TriageResult(
                esi_score=3,
                esi_rationale="stub",
                clinical_reasoning="stub",
                recommended_interventions=[],
            ),
            escalation=EscalationAssessment(level=EscalationLevel.NONE, findings=[], reasons=[]),
        )

    monkeypatch.setattr(triage_service, "run_triage", _fake)


class TestRateLimiterUnit:
    def test_allows_up_to_the_limit(self):
        clock = iter([0.0, 0.0, 0.0]).__next__
        limiter = RateLimiter(limit=2, window_seconds=60, clock=clock)
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False

    def test_keys_are_independent(self):
        limiter = RateLimiter(limit=1, window_seconds=60, clock=lambda: 0.0)
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True
        assert limiter.allow("a") is False

    def test_old_hits_expire_out_of_the_window(self):
        times = iter([0.0, 61.0, 61.0])
        limiter = RateLimiter(limit=1, window_seconds=60, clock=lambda: next(times))
        assert limiter.allow("a") is True  # hit recorded at t=0
        assert limiter.allow("a") is True  # t=61, the t=0 hit is outside the window
        # at this point two hits so far at t=0(expired) and t=61; next check also at t=61
        # third call re-checks at t=61 again, now within window of the t=61 hit above
        assert limiter.allow("a") is False

    def test_reset_clears_all_buckets(self):
        limiter = RateLimiter(limit=1, window_seconds=60, clock=lambda: 0.0)
        limiter.allow("a")
        limiter.reset()
        assert limiter.allow("a") is True


class TestCheckInRateLimitIntegration:
    def test_429_after_limit_exceeded_from_same_ip(self, client, db_session, monkeypatch):
        monkeypatch.setattr(check_in_limiter, "_limit", 2)
        make_user(db_session, "nurse1", UserRole.NURSE)
        headers = auth_headers(client, "nurse1")
        _stub_run_triage(monkeypatch)

        r1 = client.post("/encounters/check-in", headers=headers, json=_check_in_payload("A"))
        r2 = client.post("/encounters/check-in", headers=headers, json=_check_in_payload("B"))
        r3 = client.post("/encounters/check-in", headers=headers, json=_check_in_payload("C"))

        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r3.status_code == 429
        assert "Too many check-ins" in r3.json()["detail"]

    def test_different_client_ips_are_not_shared(self, client, db_session, monkeypatch):
        monkeypatch.setattr(check_in_limiter, "_limit", 1)
        make_user(db_session, "nurse1", UserRole.NURSE)
        headers = auth_headers(client, "nurse1")
        _stub_run_triage(monkeypatch)

        r1 = client.post(
            "/encounters/check-in", headers={**headers, "X-Forwarded-For": "1.1.1.1"}, json=_check_in_payload("A")
        )
        r2 = client.post(
            "/encounters/check-in", headers={**headers, "X-Forwarded-For": "2.2.2.2"}, json=_check_in_payload("B")
        )

        assert r1.status_code == 201
        assert r2.status_code == 201

    def test_other_endpoints_are_not_rate_limited(self, client, db_session, monkeypatch):
        monkeypatch.setattr(check_in_limiter, "_limit", 1)
        make_user(db_session, "nurse1", UserRole.NURSE)
        headers = auth_headers(client, "nurse1")
        _stub_run_triage(monkeypatch)

        client.post("/encounters/check-in", headers=headers, json=_check_in_payload("A"))

        for _ in range(5):
            r = client.get("/encounters", headers=headers)
            assert r.status_code == 200
