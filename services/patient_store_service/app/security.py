import os
from typing import Any, Dict, List

from fastapi import Header, HTTPException, status

try:
    from jose import jwt, JWTError
except Exception:  # pragma: no cover
    jwt = None
    JWTError = Exception

_API_KEY = os.getenv("API_KEY", "").strip()

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALG = os.getenv("JWT_ALG", "HS256").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "cse400-auth").strip()

def _decode_bearer(authorization: str) -> Dict[str, Any]:
    if jwt is None:
        raise HTTPException(status_code=500, detail="python-jose is not installed for JWT validation.")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"verify_aud": False})
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
    iss = claims.get("iss")
    if iss and iss != JWT_ISSUER:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer.")
    return claims

def require_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
    authorization: str = Header(default="", alias="Authorization"),
) -> Dict[str, Any]:
    """
    Backward-compatible auth dependency:
      - If X-API-Key matches (internal service call), allow.
      - Else require a valid JWT Bearer token (end-user call).
    """
    if _API_KEY and x_api_key == _API_KEY:
        return {"mode": "api_key", "sub": "service", "roles": ["service"]}
    return _decode_bearer(authorization)

def actor_from_headers(
    x_user_id: str = Header(default="", alias="X-User-Id"),
    authorization: str = Header(default="", alias="Authorization"),
) -> str:
    """
    Prefer JWT 'sub' when present; fall back to X-User-Id for internal service calls.
    """
    if authorization and authorization.lower().startswith("bearer "):
        try:
            claims = _decode_bearer(authorization)
            sub = claims.get("sub")
            if sub:
                return str(sub)
        except Exception:
            pass
    return x_user_id.strip() or "unknown"


def require_roles(claims: Dict[str, Any], allowed: List[str]) -> None:
    roles = claims.get("roles") or []
    if not any(role in roles for role in allowed):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role.")
