"""backend/api/routes/auth.py — Login and account creation."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user, require_role
from backend.core.security import create_access_token, hash_password, verify_password
from backend.db.session import get_db
from backend.models.audit_log import AuditAction, AuditLog
from backend.models.user import User, UserRole
from backend.schemas.auth import Token, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.username == form.username).first()

    # Same error for "no such user" and "wrong password" — distinguishing
    # them lets an attacker enumerate valid usernames.
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    db.add(AuditLog(
        actor_user_id=user.id,
        action=AuditAction.LOGIN,
        resource_type="user",
        resource_id=user.id,
    ))
    db.commit()

    token = create_access_token(subject=user.id, role=user.role.value)
    return Token(access_token=token)


@router.get("/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    # Account creation is admin-only — the audit log's promise that every
    # actor is a real, accountable person only holds if account creation
    # itself is gated, not open to whoever hits the endpoint first.
    admin: User = Depends(require_role(UserRole.ADMIN)),
) -> User:
    if db.query(User).filter(User.username == payload.username).first() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already taken")

    user = User(
        username=payload.username,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    db.add(user)
    db.flush()  # populate user.id before the audit row references it

    audit_row = AuditLog(
        actor_user_id=admin.id,
        action=AuditAction.USER_CREATED,
        resource_type="user",
        resource_id=user.id,
    )
    audit_row.metadata_dict = {"created_username": user.username, "role": user.role.value}
    db.add(audit_row)
    db.commit()
    db.refresh(user)
    return user
