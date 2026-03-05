import pytest
from fastapi import HTTPException

from services.inference_service.app import security


def test_require_api_key_accepts_matching_key(monkeypatch):
    monkeypatch.setattr(security, "_API_KEY", "k")
    claims = security.require_api_key(x_api_key="k", authorization="")
    assert claims["mode"] == "api_key"


def test_decode_bearer_requires_header_prefix():
    with pytest.raises(HTTPException) as exc:
        security._decode_bearer("token")
    assert exc.value.status_code == 401
