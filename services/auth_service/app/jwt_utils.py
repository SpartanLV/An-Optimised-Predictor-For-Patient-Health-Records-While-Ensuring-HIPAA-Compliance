import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from jose import jwt

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALG = os.getenv("JWT_ALG", "HS256").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "cse400-auth").strip()
JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS", "3600"))

def create_access_token(sub: str, roles: List[str], extra: Optional[Dict[str, Any]] = None) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not set.")
    now = datetime.now(tz=timezone.utc)
    payload: Dict[str, Any] = {
        "iss": JWT_ISSUER,
        "sub": sub,
        "roles": roles,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=JWT_EXP_SECONDS)).timestamp()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
