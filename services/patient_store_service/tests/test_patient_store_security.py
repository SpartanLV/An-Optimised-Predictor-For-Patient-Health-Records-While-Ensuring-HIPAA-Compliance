import pytest
from fastapi import HTTPException

from services.patient_store_service.app import security


def test_require_api_key_accepts_matching_key(monkeypatch):
    monkeypatch.setattr(security, "_API_KEY", "secret")

    claims = security.require_api_key(x_api_key="secret", authorization="")
    assert claims["mode"] == "api_key"
    assert claims["roles"] == ["service"]


def test_require_roles_rejects_missing_role():
    with pytest.raises(HTTPException) as exc:
        security.require_roles({"roles": ["auditor"]}, allowed=["admin", "clinician"])
    assert exc.value.status_code == 403


def test_can_view_phi_by_role(monkeypatch):
    monkeypatch.setattr(security, "PHI_UNMASK_ROLES", {"admin", "clinician"})

    assert security.can_view_phi({"roles": ["admin"]}) is True
    assert security.can_view_phi({"roles": ["auditor"]}) is False
