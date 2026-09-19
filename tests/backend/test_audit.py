"""
tests/backend/test_audit.py — The read-only admin audit trail
(backend/api/routes/audit.py), the piece of V2_PLAN.md Phase 3 that wasn't
already covered by every mutating route writing its own row.
"""

from backend.models.user import UserRole
from tests.backend.conftest import auth_headers, make_user


class TestReadAuditLog:

    def test_admin_can_read_it(self, client, db_session):
        admin = make_user(db_session, "admin1", UserRole.ADMIN)
        headers = auth_headers(client, "admin1")

        r = client.get("/audit", headers=headers)
        assert r.status_code == 200
        # Logging in as admin1 itself writes a LOGIN row.
        rows = r.json()
        assert len(rows) >= 1
        assert rows[0]["actor_username"] == "admin1"
        assert rows[0]["action"] == "login"

    def test_nurse_cannot_read_it(self, client, db_session):
        make_user(db_session, "nurse1", UserRole.NURSE)
        headers = auth_headers(client, "nurse1")
        r = client.get("/audit", headers=headers)
        assert r.status_code == 403

    def test_physician_cannot_read_it(self, client, db_session):
        make_user(db_session, "physician1", UserRole.PHYSICIAN)
        headers = auth_headers(client, "physician1")
        r = client.get("/audit", headers=headers)
        assert r.status_code == 403

    def test_filters_by_action(self, client, db_session):
        admin = make_user(db_session, "admin2", UserRole.ADMIN)
        headers = auth_headers(client, "admin2")
        make_user(db_session, "another_nurse", UserRole.NURSE)  # no login yet

        r = client.get("/audit", headers=headers, params={"action": "user_created"})
        assert r.status_code == 200
        assert all(row["action"] == "user_created" for row in r.json())
        assert len(r.json()) == 0  # no user was created through the API in this test

    def test_filters_by_resource_id(self, client, db_session):
        admin = make_user(db_session, "admin3", UserRole.ADMIN)
        headers = auth_headers(client, "admin3")

        r = client.get("/audit", headers=headers, params={"resource_id": admin.id})
        assert r.status_code == 200
        assert all(row["resource_id"] == admin.id for row in r.json())
        assert len(r.json()) >= 1  # admin3's own login row references their own user id

    def test_newest_first(self, client, db_session):
        make_user(db_session, "admin4", UserRole.ADMIN)
        make_user(db_session, "nurse2", UserRole.NURSE)
        headers_admin = auth_headers(client, "admin4")
        auth_headers(client, "nurse2")  # a second, later LOGIN row

        r = client.get("/audit", headers=headers_admin)
        timestamps = [row["timestamp"] for row in r.json()]
        assert timestamps == sorted(timestamps, reverse=True)
