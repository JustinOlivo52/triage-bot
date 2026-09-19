"""
backend/api/deps.py — Shared FastAPI dependencies: DB session, current user,
role enforcement.

Role checks live here, as a dependency, rather than as `if` statements
scattered through route bodies — one place to read and audit for
correctness, and a route simply cannot forget to enforce access, because the
dependency is part of its signature.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.core.security import decode_access_token
from backend.db.session import get_db
from backend.models.user import User, UserRole

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str = Depends(_oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    payload = decode_access_token(token)
    if payload is None:
        raise _CREDENTIALS_EXCEPTION

    user_id = payload.get("sub")
    if user_id is None:
        raise _CREDENTIALS_EXCEPTION

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _CREDENTIALS_EXCEPTION

    return user


def require_role(*allowed: UserRole):
    """
    Returns a dependency that only allows the given roles through.

    Usage: `current_user: User = Depends(require_role(UserRole.ADMIN))`.
    A role hierarchy (physician can do what a nurse can) is expressed by
    listing every role a route accepts, not by ordinal comparison — explicit
    over implicit, since "physician > nurse" is not a fact the codebase
    should quietly assume everywhere.
    """
    def _dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role.value}' cannot perform this action",
            )
        return current_user

    return _dependency
