"""
tests/backend/test_auth.py — Login, token issuance, and role enforcement
through the real HTTP layer (FastAPI TestClient), not by calling functions
directly. This is what actually proves the dependency wiring is correct.
"""

from backend.models.user import UserRole
from tests.backend.conftest import auth_headers, make_user


class TestLogin:

    def test_correct_credentials_return_a_token(self, client, db_session):
        make_user(db_session, "nurse1", UserRole.NURSE)
        r = client.post("/auth/login", data={"username": "nurse1", "password": "testpass123"})
        assert r.status_code == 200
        assert r.json()["token_type"] == "bearer"
        assert r.json()["access_token"]

    def test_wrong_password_is_rejected(self, client, db_session):
        make_user(db_session, "nurse2", UserRole.NURSE)
        r = client.post("/auth/login", data={"username": "nurse2", "password": "wrong"})
        assert r.status_code == 401

    def test_unknown_username_is_rejected(self, client, db_session):
        r = client.post("/auth/login", data={"username": "ghost", "password": "x"})
        assert r.status_code == 401

    def test_wrong_password_and_unknown_username_give_the_same_error(self, client, db_session):
        """Must not let a caller distinguish 'no such user' from 'wrong password'."""
        make_user(db_session, "nurse3", UserRole.NURSE)
        r1 = client.post("/auth/login", data={"username": "nurse3", "password": "wrong"})
        r2 = client.post("/auth/login", data={"username": "ghost2", "password": "wrong"})
        assert r1.status_code == r2.status_code == 401
        assert r1.json()["detail"] == r2.json()["detail"]

    def test_disabled_account_cannot_log_in(self, client, db_session):
        user = make_user(db_session, "disabled1", UserRole.NURSE)
        user.is_active = False
        db_session.commit()

        r = client.post("/auth/login", data={"username": "disabled1", "password": "testpass123"})
        assert r.status_code == 403


class TestMe:

    def test_returns_the_authenticated_user(self, client, db_session):
        make_user(db_session, "nurse4", UserRole.NURSE)
        headers = auth_headers(client, "nurse4")

        r = client.get("/auth/me", headers=headers)
        assert r.status_code == 200
        assert r.json()["username"] == "nurse4"
        assert r.json()["role"] == "nurse"

    def test_rejects_no_token(self, client, db_session):
        r = client.get("/auth/me")
        assert r.status_code == 401

    def test_rejects_garbage_token(self, client, db_session):
        r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        assert r.status_code == 401


class TestUserCreation:
    """Account creation is admin-only — this is the role-enforcement test
    that matters most, since it gates who can become an accountable actor
    in the audit log at all."""

    def test_admin_can_create_a_user(self, client, db_session):
        make_user(db_session, "admin1", UserRole.ADMIN)
        headers = auth_headers(client, "admin1")

        r = client.post("/auth/users", headers=headers, json={
            "username": "newnurse", "password": "newpass123",
            "full_name": "New Nurse", "role": "nurse",
        })
        assert r.status_code == 201
        assert r.json()["username"] == "newnurse"
        assert "password" not in r.json()  # never echo the password back

    def test_nurse_cannot_create_a_user(self, client, db_session):
        make_user(db_session, "nurse5", UserRole.NURSE)
        headers = auth_headers(client, "nurse5")

        r = client.post("/auth/users", headers=headers, json={
            "username": "x", "password": "password123", "full_name": "X", "role": "nurse",
        })
        assert r.status_code == 403

    def test_physician_cannot_create_a_user(self, client, db_session):
        """Physician has more clinical authority than a nurse, but account
        creation is an admin action specifically — the two are not the same
        axis of privilege."""
        make_user(db_session, "physician1", UserRole.PHYSICIAN)
        headers = auth_headers(client, "physician1")

        r = client.post("/auth/users", headers=headers, json={
            "username": "x", "password": "password123", "full_name": "X", "role": "nurse",
        })
        assert r.status_code == 403

    def test_duplicate_username_is_rejected(self, client, db_session):
        make_user(db_session, "admin2", UserRole.ADMIN)
        make_user(db_session, "taken", UserRole.NURSE)
        headers = auth_headers(client, "admin2")

        r = client.post("/auth/users", headers=headers, json={
            "username": "taken", "password": "password123", "full_name": "X", "role": "nurse",
        })
        assert r.status_code == 409

    def test_creating_a_user_writes_an_audit_row(self, client, db_session):
        admin = make_user(db_session, "admin3", UserRole.ADMIN)
        headers = auth_headers(client, "admin3")

        client.post("/auth/users", headers=headers, json={
            "username": "audited", "password": "password123", "full_name": "X", "role": "nurse",
        })

        from backend.models.audit_log import AuditAction, AuditLog
        rows = db_session.query(AuditLog).filter(AuditLog.action == AuditAction.USER_CREATED).all()
        assert len(rows) == 1
        assert rows[0].actor_user_id == admin.id
        assert rows[0].metadata_dict["created_username"] == "audited"


class TestPasswordHashing:

    def test_password_is_never_stored_in_plaintext(self, db_session):
        user = make_user(db_session, "hashcheck", UserRole.NURSE, password="supersecret123")
        assert user.hashed_password != "supersecret123"
        assert user.hashed_password.startswith("$2b$")  # bcrypt's own prefix
