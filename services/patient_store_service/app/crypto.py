import os
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken

_KEY = os.getenv("FILE_ENCRYPTION_KEY", "").strip()

_fernet: Optional[Fernet] = None
if _KEY:
    try:
        _fernet = Fernet(_KEY.encode("utf-8"))
    except Exception:
        _fernet = None

def encryption_enabled() -> bool:
    return _fernet is not None

def encrypt(data: bytes) -> bytes:
    if _fernet is None:
        return data
    return _fernet.encrypt(data)

def decrypt(data: bytes) -> bytes:
    if _fernet is None:
        return data
    try:
        return _fernet.decrypt(data)
    except InvalidToken:
        # If key changed or corrupted, return raw (caller can decide)
        return data
