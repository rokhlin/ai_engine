import pytest
import os
import io
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app
from src import config, database, workers
from src.utils.media_loader import resolve_or_stream_media_file


@pytest.fixture(autouse=True)
def setup_test_env():
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)

    old_output = config.OUTPUT_FOLDER
    old_faces = config.FACES_FOLDER
    old_db = config.DB_PATH

    config.OUTPUT_FOLDER = temp_path
    config.FACES_FOLDER = temp_path / "facess"
    config.DB_PATH = temp_path / "test_stateless.db"

    config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    config.FACES_FOLDER.mkdir(parents=True, exist_ok=True)
    database.init_db()

    yield

    config.OUTPUT_FOLDER = old_output
    config.FACES_FOLDER = old_faces
    config.DB_PATH = old_db
    try:
        temp_dir.cleanup()
    except Exception:
        pass


def test_resolve_or_stream_media_file_local_accessible(tmp_path):
    local_file = tmp_path / "sample.jpg"
    local_file.write_bytes(b"dummy image data")

    resolved = resolve_or_stream_media_file(str(local_file))
    assert resolved == str(local_file)


def test_resolve_or_stream_media_file_stream_fallback():
    fake_remote_path = "Z:\\RemotePhotos\\vacation.jpg"
    fake_stream_url = "http://localhost:8000/api/media/file?path=Z%3A%5CRemotePhotos%5Cvacation.jpg"

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"downloaded image bytes from stream url", b""]
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=mock_resp):
        temp_downloaded_path = resolve_or_stream_media_file(
            fake_remote_path,
            stream_url=fake_stream_url
        )
        assert os.path.isfile(temp_downloaded_path)
        assert Path(temp_downloaded_path).suffix == ".jpg"
        with open(temp_downloaded_path, "rb") as f:
            assert f.read() == b"downloaded image bytes from stream url"

        # Cleanup
        os.unlink(temp_downloaded_path)


def test_resolve_or_stream_media_file_ui_base_url_fallback():
    fake_remote_path = "/nas/photos/img001.png"
    ui_base_url = "http://192.168.1.100:8000"

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [b"png bytes", b""]
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        temp_path = resolve_or_stream_media_file(
            fake_remote_path,
            ui_base_url=ui_base_url
        )
        assert os.path.isfile(temp_path)
        req = mock_urlopen.call_args[0][0]
        assert "192.168.1.100:8000/api/media/file" in req.full_url
        os.unlink(temp_path)


def test_resolve_or_stream_media_file_missing_raises():
    fake_remote_path = "Z:\\Missing\\not_found.jpg"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_or_stream_media_file(fake_remote_path)
    assert "not accessible" in str(exc_info.value)


def test_api_run_sync_with_payload():
    client = TestClient(app)

    payload = {
        "force": True,
        "total_files": 2,
        "files": [
            {
                "file_path": "Z:\\Photo\\vacation.jpg",
                "folder": "Z:\\Photo",
                "filename": "vacation.jpg",
                "file_size": 2048,
                "mtime": 1740000000.0,
                "stream_url": "http://localhost:8000/api/media/file?path=Z%3A%5CPhoto%5Cvacation.jpg"
            },
            {
                "file_path": "Z:\\Photo\\family.jpg",
                "folder": "Z:\\Photo",
                "filename": "family.jpg",
                "file_size": 4096,
                "mtime": 1740000010.0,
                "stream_url": "http://localhost:8000/api/media/file?path=Z%3A%5CPhoto%5Cfamily.jpg"
            }
        ],
        "output_folder": str(config.OUTPUT_FOLDER),
        "settings": {
            "output_folder": str(config.OUTPUT_FOLDER),
            "model_provider": "gemini",
            "gemini_model": "gemini-3.6-flash",
            "gemini_max_workers": 3,
            "whisper_model": "large-v3-turbo",
            "preserve_structure": True,
            "ui_base_url": "http://localhost:8000"
        }
    }

    with patch("api.run_pipeline_thread", return_value=None):
        response = client.post("/api/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["total_files"] == 2
        assert "2 provided files" in data["message"]


def test_api_analyze_file_with_payload():
    client = TestClient(app)

    payload = {
        "file": "Z:\\Photo\\portrait.jpg",
        "filename": "portrait.jpg",
        "folder": "Z:\\Photo",
        "file_size": 1024,
        "mtime": 1740000000.0,
        "output_folder": str(config.OUTPUT_FOLDER),
        "stream_url": "http://localhost:8000/api/media/file?path=Z%3A%5CPhoto%5Cportrait.jpg",
        "settings": {
            "model_provider": "local",
            "local_model_name": "qwen2.5-vl-7b-instruct",
            "ui_base_url": "http://localhost:8000"
        }
    }

    with patch("api.run_analyze_file_thread", return_value=None):
        response = client.post("/api/analyze-file", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["file"] == "Z:\\Photo\\portrait.jpg"
        assert "portrait.jpg" in data["message"]


def test_process_photo_resilient_error_handling(tmp_path):
    """Test that process_photo records FAILED status in DB and does not crash when file is inaccessible."""
    missing_item = {
        "file_path": "Z:\\Inaccessible\\ghost.jpg",
        "folder": "Z:\\Inaccessible",
        "filename": "ghost.jpg",
        "file_size": 12345,
        "mtime": 1740000000.0,
        "stream_url": None
    }

    result = workers.process_photo(missing_item)
    assert result is None

    # Check database status
    rec = database.get_sync_record("Z:\\Inaccessible\\ghost.jpg")
    assert rec is not None
    assert rec["status"] == "FAILED"
    assert "not accessible" in rec["error_message"] or "Failed to stream" in rec["error_message"]

