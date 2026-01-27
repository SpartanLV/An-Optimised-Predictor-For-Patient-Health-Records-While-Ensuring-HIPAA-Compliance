from typing import Any, Dict, List

import uuid

from fastapi import FastAPI, Depends, HTTPException, status, Header, Request
from fastapi.security import OAuth2PasswordRequestForm

from .user_store import find_user, verify_password, upsert_user
from .jwt_utils import create_access_token, JWT_EXP_SECONDS
from .schemas import TokenResponse, CreateUserRequest, UserOut
from .security import decode_bearer_token, require_roles

APP_NAME = "cse400-auth-service"

app = FastAPI(
    title="CSE400 Auth Service",
    version="0.1.0",
    description="Minimal auth service for demo: issues JWTs for the web platform.",
)

@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME}

@app.post("/v1/auth/token", response_model=TokenResponse)
def token(form: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 password flow (demo only). In real deployments, integrate OIDC provider (Keycloak/Okta/Azure AD).
    """
    u = find_user(form.username)
    if not u or not verify_password(form.password, u.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")

    roles: List[str] = u.get("roles") or []
    access_token = create_access_token(sub=form.username, roles=roles)

    return TokenResponse(
        access_token=access_token,
        expires_in=int(JWT_EXP_SECONDS),
        roles=roles,
    )

def get_claims(authorization: str = Header(default="", alias="Authorization")) -> Dict[str, Any]:
    return decode_bearer_token(authorization)

@app.get("/v1/auth/me", response_model=UserOut)
def me(claims: Dict[str, Any] = Depends(get_claims)):
    return UserOut(username=claims.get("sub", "unknown"), roles=claims.get("roles") or [])

@app.post("/v1/auth/users", response_model=UserOut)
def create_user(req: CreateUserRequest, claims: Dict[str, Any] = Depends(get_claims)):
    # Admin-only endpoint
    require_roles(claims, allowed=["admin"])
    u = upsert_user(req.username, req.password, req.roles)
    return UserOut(username=u["username"], roles=u.get("roles") or [])
