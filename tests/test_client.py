import pytest
import requests
from unittest.mock import MagicMock, patch
from media_cataloger_client import (
    MediaCatalogerClient,
    AuthenticationError,
    PermissionDeniedError,
    InvalidRequestError,
    CatalogerApiError
)

BASE_URL = "http://localhost:8000"


def _create_mock_response(status_code=200, json_data=None, text=""):
    mock = MagicMock()
    mock.status_code = status_code
    mock.ok = 200 <= status_code < 300
    mock.json.return_value = json_data or {}
    mock.text = text or str(json_data)
    return mock


def test_client_login_success():
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = _create_mock_response(
        status_code=200,
        json_data={
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test_payload.test_sig",
            "user": {
                "id": "user_admin_default",
                "username": "admin",
                "displayName": "Administrator",
                "role": "admin",
                "permissions": ["view_media", "edit_metadata", "manage_faces", "admin_panel", "vault_access"]
            }
        }
    )

    client = MediaCatalogerClient(base_url=BASE_URL, session=mock_session)
    user = client.login("admin", "admin_pass")
    assert user["username"] == "admin"
    assert client.token is not None
    assert client.session.headers["Authorization"] == f"Bearer {client.token}"
    mock_session.post.assert_called_once()


def test_client_login_invalid_credentials():
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = _create_mock_response(
        status_code=401,
        json_data={"detail": "Invalid username or password"}
    )

    client = MediaCatalogerClient(base_url=BASE_URL, session=mock_session)
    with pytest.raises(AuthenticationError):
        client.login("admin", "wrong")


def test_client_unlock_and_lock_vault():
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.side_effect = [
        _create_mock_response(
            status_code=200,
            json_data={"unlocked": True, "sessionToken": "vault_sess_12345", "expiresAt": 1788265800000}
        ),
        _create_mock_response(
            status_code=200,
            json_data={"unlocked": False, "message": "Vault locked"}
        )
    ]

    client = MediaCatalogerClient(base_url=BASE_URL, session=mock_session)
    vtoken = client.unlock_vault("1234")
    assert vtoken == "vault_sess_12345"
    assert client.session.headers["x-vault-token"] == "vault_sess_12345"

    client.lock_vault()
    assert client.vault_token is None
    assert "x-vault-token" not in client.session.headers


def test_client_save_metadata():
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.request.return_value = _create_mock_response(
        status_code=200,
        json_data={"status": "success", "message": "Metadata persisted successfully."}
    )

    client = MediaCatalogerClient(base_url=BASE_URL, session=mock_session)
    result = client.save_metadata(
        "/media/photos/sample.jpg",
        {"summary": "Test summary", "tags": ["nature"]}
    )
    assert result["status"] == "success"
    mock_session.request.assert_called_with(
        "POST",
        f"{BASE_URL}/api/media/metadata",
        json={"file": "/media/photos/sample.jpg", "summary": "Test summary", "tags": ["nature"]},
        timeout=15.0
    )


def test_client_assign_face():
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.request.return_value = _create_mock_response(
        status_code=200,
        json_data={"status": "success", "message": "Face assigned successfully."}
    )

    client = MediaCatalogerClient(base_url=BASE_URL, session=mock_session)
    result = client.assign_face("/media/photos/sample.jpg", "face_001", "Alexander", confidence=0.95)
    assert result["status"] == "success"
    mock_session.request.assert_called_with(
        "POST",
        f"{BASE_URL}/api/faces/assign",
        json={
            "file": "/media/photos/sample.jpg",
            "face_id": "face_001",
            "person_name": "Alexander",
            "confidence": 0.95
        },
        timeout=15.0
    )


def test_client_trigger_sync_and_status():
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.request.side_effect = [
        _create_mock_response(
            status_code=200,
            json_data={"status": "started", "message": "Cataloging started"}
        ),
        _create_mock_response(
            status_code=200,
            json_data={"status": "running", "current_task": "sync"}
        )
    ]

    client = MediaCatalogerClient(base_url=BASE_URL, session=mock_session)
    sync_res = client.trigger_full_sync(force=False)
    assert sync_res["status"] == "started"

    status_res = client.get_status()
    assert status_res["status"] == "running"


def test_client_auto_relogin_on_401():
    mock_session = MagicMock()
    mock_session.headers = {}
    
    # Sequence:
    # 1. request -> 401 Unauthorized
    # 2. post(/api/auth/login) -> 200 OK with new token
    # 3. retry request -> 200 OK
    mock_session.request.side_effect = [
        _create_mock_response(status_code=401, json_data={"detail": "Token expired"}),
        _create_mock_response(status_code=200, json_data={"status": "idle"})
    ]
    mock_session.post.return_value = _create_mock_response(
        status_code=200,
        json_data={"token": "new_refreshed_token", "user": {"username": "admin"}}
    )

    client = MediaCatalogerClient(base_url=BASE_URL, username="admin", password="password123", session=mock_session)
    client.token = "old_expired_token"
    client.session.headers.update({"Authorization": "Bearer old_expired_token"})

    status = client.get_status()
    assert status["status"] == "idle"
    assert client.token == "new_refreshed_token"
    mock_session.post.assert_called_once()
