from fastapi.testclient import TestClient

from services.auth_service.app import main


client = TestClient(main.app)


def teardown_function():
    main.app.dependency_overrides.clear()


def test_token_success(monkeypatch):
    monkeypatch.setattr(main, "find_user", lambda username: {"username": username, "password_hash": "x", "roles": ["admin"]})
    monkeypatch.setattr(main, "verify_password", lambda plain, hashed: plain == "good")
    monkeypatch.setattr(main, "create_access_token", lambda sub, roles: f"token-for-{sub}-{','.join(roles)}")

    response = client.post(
        "/v1/auth/token",
        data={"username": "alice", "password": "good"},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "token-for-alice-admin"
    assert body["roles"] == ["admin"]


def test_token_invalid_credentials(monkeypatch):
    monkeypatch.setattr(main, "find_user", lambda _: None)

    response = client.post(
        "/v1/auth/token",
        data={"username": "missing", "password": "bad"},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials."


def test_me_reads_claims_from_override():
    main.app.dependency_overrides[main.get_claims] = lambda: {"sub": "demo", "roles": ["clinician"]}

    response = client.get("/v1/auth/me")
    assert response.status_code == 200
    assert response.json() == {"username": "demo", "roles": ["clinician"]}


def test_create_user_requires_admin_role(monkeypatch):
    main.app.dependency_overrides[main.get_claims] = lambda: {"sub": "auditor", "roles": ["auditor"]}

    response = client.post(
        "/v1/auth/users",
        json={"username": "new", "password": "Passw0rd!", "roles": ["clinician"]},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Insufficient role."


def test_create_user_as_admin(monkeypatch):
    main.app.dependency_overrides[main.get_claims] = lambda: {"sub": "admin", "roles": ["admin"]}
    monkeypatch.setattr(main, "upsert_user", lambda username, password, roles: {"username": username, "roles": roles})

    response = client.post(
        "/v1/auth/users",
        json={"username": "new", "password": "Passw0rd!", "roles": ["clinician"]},
    )

    assert response.status_code == 200
    assert response.json() == {"username": "new", "roles": ["clinician"]}
