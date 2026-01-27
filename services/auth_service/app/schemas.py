from typing import List, Optional
from pydantic import BaseModel, Field

class TokenResponse(BaseModel):
    token_type: str = "bearer"
    access_token: str
    expires_in: int
    roles: List[str] = []

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    roles: List[str] = ["clinician"]

class UserOut(BaseModel):
    username: str
    roles: List[str] = []
