"""backend/schemas/auth.py — Request/response shapes for the auth routes."""

from pydantic import BaseModel, Field

from backend.models.user import UserRole


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=200)
    role: UserRole


class UserOut(BaseModel):
    id: str
    username: str
    full_name: str
    role: UserRole
    is_active: bool

    model_config = {"from_attributes": True}
