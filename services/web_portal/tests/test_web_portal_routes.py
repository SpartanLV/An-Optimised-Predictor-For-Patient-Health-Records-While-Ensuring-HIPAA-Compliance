from fastapi.testclient import TestClient

from services.web_portal.app import main


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise main.httpx.HTTPStatusError(
                message="error",
                request=main.httpx.Request("GET", "http://fake"),
                response=main.httpx.Response(self.status_code, request=main.httpx.Request("GET", "http://fake")),
            )

    def json(self):
        return self._payload


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self.fail_audit = kwargs.pop("fail_audit", False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, headers=None, params=None):
        if url.endswith("/v1/patients"):
            return FakeResponse(payload=[
                {"id": "abc-1", "mrn": "M100", "first_name": "Alice", "last_name": "Ng", "created_at_utc": "2024-01-30T00:00:00Z"},
                {"id": "abc-2", "mrn": "", "first_name": "Bob", "last_name": "Li", "created_at_utc": "2023-12-01T00:00:00Z"},
            ])
        if url.endswith("/v1/audit"):
            raise main.httpx.ConnectError("down", request=main.httpx.Request("GET", url))
        if url.endswith("/health"):
            return FakeResponse(payload={"status": "ok"})
        if url.endswith("/v1/auth/me"):
            return FakeResponse(payload={"username": "admin", "roles": ["admin"]})
        return FakeResponse(payload={})

    async def post(self, *args, **kwargs):
        return FakeResponse(payload={"access_token": "x"})


def _set_auth_cookie(client: TestClient):
    client.cookies.set(main.COOKIE_NAME, "fake-token")


def test_patients_route_renders_filtered_results(monkeypatch):
    async def fake_me(_):
        return {"username": "admin", "roles": ["admin"]}

    monkeypatch.setattr(main, "_me", fake_me)
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)

    client = TestClient(main.app)
    _set_auth_cookie(client)
    resp = client.get("/patients?q=alice")
    assert resp.status_code == 200
    assert "Showing 1 result" in resp.text


def test_audit_route_shows_error_message_when_service_unavailable(monkeypatch):
    async def fake_me(_):
        return {"username": "admin", "roles": ["admin"]}

    monkeypatch.setattr(main, "_me", fake_me)
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)

    client = TestClient(main.app)
    _set_auth_cookie(client)
    resp = client.get("/audit")
    assert resp.status_code == 200
    assert "Unable to load audit logs" in resp.text


def test_system_health_route_includes_latency_and_endpoint(monkeypatch):
    async def fake_me(_):
        return {"username": "admin", "roles": ["admin"]}

    monkeypatch.setattr(main, "_me", fake_me)
    monkeypatch.setattr(main.httpx, "AsyncClient", FakeAsyncClient)

    client = TestClient(main.app)
    _set_auth_cookie(client)
    resp = client.get("/system-health")
    assert resp.status_code == 200
    assert "Latency" in resp.text
    assert "Endpoint" in resp.text


def test_api_contracts_route_renders_markdown(monkeypatch):
    async def fake_me(_):
        return {"username": "admin", "roles": ["admin"]}

    monkeypatch.setattr(main, "_me", fake_me)

    client = TestClient(main.app)
    _set_auth_cookie(client)
    resp = client.get("/api-contracts")
    assert resp.status_code == 200
    assert "API Contracts (MVP)" in resp.text
