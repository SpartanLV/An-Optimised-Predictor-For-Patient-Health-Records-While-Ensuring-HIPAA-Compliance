import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")  # avoids bcrypt 72-byte limit

USERS_PATH = Path(os.getenv("USERS_FILE", "/data/users.json")).resolve()

def _default_admin() -> Dict[str, Any]:
    """
    Creates an initial admin user if USERS_FILE does not exist.
    IMPORTANT: In real deployments, provision users via secure admin workflow.
    """
    username = os.getenv("ADMIN_USERNAME", "admin").strip()
    password = os.getenv("ADMIN_PASSWORD", "admin123").strip()  # dev default; override in .env
    roles = ["admin", "clinician"]
    return {
        "users": [
            {
                "username": username,
                "password_hash": pwd_context.hash(password),
                "roles": roles,
            }
        ]
    }

def load_users() -> Dict[str, Any]:
    if not USERS_PATH.exists():
        USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = _default_admin()
        USERS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    return json.loads(USERS_PATH.read_text(encoding="utf-8") or "{}")

def save_users(data: Dict[str, Any]) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

def find_user(username: str) -> Optional[Dict[str, Any]]:
    data = load_users()
    for u in data.get("users", []):
        if u.get("username") == username:
            return u
    return None

def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(plain_password, password_hash)
    except Exception:
        return False

def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)

def upsert_user(username: str, plain_password: str, roles: List[str]) -> Dict[str, Any]:
    data = load_users()
    users = data.get("users", [])
    for u in users:
        if u.get("username") == username:
            u["password_hash"] = hash_password(plain_password)
            u["roles"] = roles
            save_users(data)
            return u
    u = {"username": username, "password_hash": hash_password(plain_password), "roles": roles}
    users.append(u)
    data["users"] = users
    save_users(data)
    return u
