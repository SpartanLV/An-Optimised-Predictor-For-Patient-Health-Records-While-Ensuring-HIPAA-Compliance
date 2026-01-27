import os
from typing import Dict, Any, List
from fastapi import Header, HTTPException, status
from jose import jwt, JWTError

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALG = os.getenv("JWT_ALG", "HS256").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "cse400-auth").strip()

def decode_bearer_token(authorization: str) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"verify_aud": False})
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
    # Basic issuer check
    iss = claims.get("iss")
    if iss and iss != JWT_ISSUER:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token issuer.")
    return claims

def require_roles(claims: Dict[str, Any], allowed: List[str]) -> None:
    roles = claims.get("roles") or []
    if not any(r in roles for r in allowed):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role.")
