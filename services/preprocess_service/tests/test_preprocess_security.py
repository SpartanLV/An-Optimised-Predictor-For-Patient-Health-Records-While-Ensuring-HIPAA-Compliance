import pytest
from fastapi import HTTPException

from services.preprocess_service.app import security


def test_require_auth_accepts_matching_api_key(monkeypatch):
    monkeypatch.setattr(security, "_API_KEY", "k")
    claims = security.require_auth(x_api_key="k", authorization="")
    assert claims["mode"] == "api_key"


def test_decode_bearer_missing_secret(monkeypatch):
    monkeypatch.setattr(security, "JWT_SECRET", "")
    with pytest.raises(HTTPException) as exc:
        security._decode_bearer("Bearer abc")
    assert exc.value.status_code == 500
