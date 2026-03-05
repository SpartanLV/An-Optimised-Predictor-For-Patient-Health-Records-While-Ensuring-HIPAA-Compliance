import pytest
from fastapi import HTTPException

from services.auth_service.app import jwt_utils, security


def test_create_access_token_requires_secret(monkeypatch):
    monkeypatch.setattr(jwt_utils, "JWT_SECRET", "")
    with pytest.raises(RuntimeError):
        jwt_utils.create_access_token(sub="alice", roles=["admin"])


def test_require_roles_rejects_disallowed_role():
    with pytest.raises(HTTPException) as exc:
        security.require_roles({"roles": ["clinician"]}, allowed=["admin"])
    assert exc.value.status_code == 403
