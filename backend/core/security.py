"""
backend/core/security.py — Password hashing and JWT issue/verify.

Two library swaps happened here, both for the same reason: an abandoned
convenience wrapper turned out to be fragile, and the actively-maintained
library underneath it did not have the problem.

  - PyJWT instead of python-jose: jose's cryptography backend pulled in a
    native/Rust extension that failed to import in this environment.
  - bcrypt directly instead of passlib: passlib is unmaintained since ~2020
    and its version-sniffing breaks against current bcrypt releases (which
    dropped the `__about__` attribute passlib checks for) — it surfaced here
    as spurious "password cannot be longer than 72 bytes" errors on
    ordinary short passwords, not a real length problem.

Both are now one dependency doing one job, not two.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from backend.core.config import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, JWT_SECRET_KEY


def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(subject: str, role: str) -> str:
    """
    Issue a JWT. `subject` is the user's id (not username — usernames can be
    reassigned, ids should not be), `role` is embedded so route dependencies
    can enforce access without a database round-trip on every request.
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": subject, "role": role, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Returns the decoded payload, or None if the token is invalid or expired."""
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
