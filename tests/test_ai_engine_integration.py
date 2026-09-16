import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app
from src import config, database, workers
from media_cataloger_client import MediaCatalogerClient


def test_api_status_contract_compliance():
    client = TestClient(app)
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    # Section 2.1 Contract Assertions
    assert "status" in data
    assert data["connected"] is True
    assert data["version"] == config.VERSION
    assert data["engine_ready"] is True
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)
    for model_key in ("face_detection", "face_recognition", "whisper", "vision_llm"):
        assert model_key in data["models_loaded"]

    assert "device" in data
    assert "current_task" in data
    assert "current_file" in data

    # Progress schema
    assert "progress" in data
    prog = data["progress"]
    assert "current" in prog
    assert "total" in prog
    assert "percentage" in prog
    assert isinstance(prog["percentage"], (int, float))

    # Queue schema
    assert "queue" in data
    q = data["queue"]
    assert "pending_count" in q
    assert "in_flight_files" in q
    assert isinstance(q["in_flight_files"], list)


def test_api_health_contract_compliance():
    client = TestClient(app)

    # 1. New /api/health endpoint
    res_api_health = client.get("/api/health")
    assert res_api_health.status_code == 200
    data = res_api_health.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert isinstance(data["timestamp"], (int, float))
    assert "models_ready" in data

    # 2. Legacy /health endpoint backwards compatibility
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_legacy = res_health.json()
    assert data_legacy["status"] == "ok"
    assert data_legacy["service"] == "media_cataloger"


def test_api_analyze_file_integration_contract():
    client = TestClient(app)

    payload = {
        "file": "C:/Photos/2026/family_vacation.jpg",
        "filename": "family_vacation.jpg",
        "folder": "C:/Photos/2026",
        "file_size": 4194304,
        "mtime": 1788340000.0,
        "output_folder": str(config.OUTPUT_FOLDER),
        "stream_url": "http://web-gateway:8000/api/media/file?path=C%3A%2FPhotos%2F2026%2Ffamily_vacation.jpg",
        "settings": {
            "llm_provider": "gemini",
            "enable_face_recognition": True,
            "enable_transcription": True
        }
    }

    with patch("api.run_analyze_file_thread", return_value=None):
        response = client.post("/api/analyze-file", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["file"] == "C:/Photos/2026/family_vacation.jpg"
        assert "family_vacation.jpg" in data["message"]


def test_client_sdk_health_and_status():
    # Test client using mocked requests
    sdk_client = MediaCatalogerClient(base_url="http://mock-cataloger:8001")

    mock_resp_health = MagicMock()
    mock_resp_health.status_code = 200
    mock_resp_health.headers = {"content-type": "application/json"}
    mock_resp_health.json.return_value = {
        "status": "healthy",
        "timestamp": time.time(),
        "models_ready": True
    }

    mock_resp_status = MagicMock()
    mock_resp_status.status_code = 200
    mock_resp_status.headers = {"content-type": "application/json"}
    mock_resp_status.json.return_value = {
        "status": "idle",
        "connected": True,
        "version": "2.4.0",
        "engine_ready": True
    }

    with patch.object(sdk_client.session, "request") as mock_req:
        mock_req.side_effect = [mock_resp_health, mock_resp_status]

        health_res = sdk_client.health()
        assert health_res["status"] == "healthy"

        status_res = sdk_client.get_status()
        assert status_res["connected"] is True
        assert status_res["version"] == "2.4.0"
