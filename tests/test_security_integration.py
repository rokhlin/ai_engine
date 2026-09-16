import pytest
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient

from api import app, create_jwt_token, decode_and_verify_token, _active_vault_sessions
from src import config, database, workers

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    out_dir = tmp_path / "output"
    in_dir = tmp_path / "input"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)

    config.OUTPUT_FOLDER = out_dir
    config.INPUT_FOLDERS = [in_dir]
    config.DB_PATH = out_dir / "catalog_history.db"
    config.FACES_FOLDER = out_dir / "facess"
    config.FACES_FOLDER.mkdir(parents=True, exist_ok=True)
    config.AI_SERVICE_USER = "admin"
    config.AI_SERVICE_PASSWORD = "admin_secure_password"
    config.VAULT_MASTER_PIN = "9876"

    database.init_db()
    _active_vault_sessions.clear()
    yield


def test_auth_login_success():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin_secure_password"})
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["user"]["username"] == "admin"
    assert "edit_metadata" in data["user"]["permissions"]
    assert "admin_panel" in data["user"]["permissions"]
    assert "manage_faces" in data["user"]["permissions"]

    # Verify token decoding
    payload = decode_and_verify_token(data["token"])
    assert payload is not None
    assert payload["username"] == "admin"


def test_auth_login_failure():
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "wrong_password"})
    assert resp.status_code == 401


def test_vault_unlock_and_lock():
    # Attempt unlock with wrong PIN
    resp = client.post("/api/vault/unlock", json={"pin": "0000"})
    assert resp.status_code == 401

    # Unlock with valid PIN
    resp = client.post("/api/vault/unlock", json={"pin": "9876"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["unlocked"] is True
    session_token = data["sessionToken"]
    assert session_token.startswith("vault_session_")

    # Check status with header
    status_resp = client.get("/api/vault/status", headers={"x-vault-token": session_token})
    assert status_resp.status_code == 200
    assert status_resp.json()["unlocked"] is True

    # Lock vault
    lock_resp = client.post("/api/vault/lock", headers={"x-vault-token": session_token})
    assert lock_resp.status_code == 200
    assert lock_resp.json()["unlocked"] is False

    # Check status again -> locked
    status_resp2 = client.get("/api/vault/status", headers={"x-vault-token": session_token})
    assert status_resp2.json()["unlocked"] is False


def test_save_media_metadata_endpoint(tmp_path):
    photo_file = config.INPUT_FOLDERS[0] / "photo_summer.jpg"
    photo_file.write_bytes(b"dummy image bytes")

    payload = {
        "file": str(photo_file),
        "summary": "Summer beach vacation",
        "summary_ru": "Летний отдых на пляже",
        "description": "Family walking along sandy coast during sunset",
        "description_ru": "Семья гуляет по песчаному берегу на закате",
        "environment": "outdoor",
        "lighting": "golden hour",
        "weather": "sunny",
        "time_of_day": "sunset",
        "tags": ["beach", "vacation", "sea"],
        "transcription": "Listen to the waves...",
        "ocr_text": "Resort Entrance"
    }

    resp = client.post("/api/media/metadata", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"

    # Verify persisted in database
    db_metadata = database.get_all_media_metadata()
    record = db_metadata.get(str(photo_file)) or db_metadata.get(photo_file.name.lower())
    assert record is not None
    assert record["summary"] == "Summer beach vacation"
    assert record["summary_ru"] == "Летний отдых на пляже"
    assert record["environment"] == "outdoor"

    # Verify sidecar JSON file created on disk
    sidecar_path = config.OUTPUT_FOLDER / f"{photo_file.name}.json"
    assert sidecar_path.is_file()
    with open(sidecar_path, "r", encoding="utf-8") as sf:
        sidecar_data = json.load(sf)
        assert sidecar_data["summary"] == "Summer beach vacation"
        assert sidecar_data["tags"] == ["beach", "vacation", "sea"]


def test_assign_face_with_person_name(tmp_path):
    import numpy as np
    # Register an unrecognized face in DB first
    face_id = "face_101"
    database.save_face_to_registry(
        face_id=face_id,
        name="face_101",
        embedding=np.zeros(512, dtype=np.float32),
        is_reference=False
    )


    assign_payload = {
        "file": str(config.INPUT_FOLDERS[0] / "family.jpg"),
        "face_id": face_id,
        "person_name": "Alexander",
        "confidence": 0.96
    }
    resp = client.post("/api/faces/assign", json=assign_payload)
    assert resp.status_code == 200

    mapping = database.get_face_name_mapping()
    assert mapping[face_id] == "Alexander"


def test_vault_isolation_in_media_files_listing(tmp_path):
    # Create standard photo
    std_photo = config.INPUT_FOLDERS[0] / "standard_pic.jpg"
    std_photo.write_bytes(b"standard")

    # Create vault photo in .vault folder
    vault_dir = config.INPUT_FOLDERS[0] / ".vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    vault_photo = vault_dir / "secret_pic.jpg"
    vault_photo.write_bytes(b"secret")

    # 1. Standard listing without vault
    resp = client.get("/api/media/files")
    assert resp.status_code == 200
    files = [f["filename"] for f in resp.json()["files"]]
    assert "standard_pic.jpg" in files
    assert "secret_pic.jpg" not in files

    # 2. Unlock vault and list with vault=true and header
    unlock_resp = client.post("/api/vault/unlock", json={"pin": "9876"})
    vtoken = unlock_resp.json()["sessionToken"]

    resp_vault = client.get("/api/media/files?vault=true", headers={"x-vault-token": vtoken})
    assert resp_vault.status_code == 200
    vault_files = [f["filename"] for f in resp_vault.json()["files"]]
    assert "standard_pic.jpg" in vault_files
    assert "secret_pic.jpg" in vault_files
