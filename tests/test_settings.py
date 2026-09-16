import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api import app
from src import config

@pytest.fixture(autouse=True)
def clean_settings():
    # Save original settings
    original_input_folders = list(config.INPUT_FOLDERS)
    original_output_folder = config.OUTPUT_FOLDER
    original_db_path = config.DB_PATH
    
    settings_path = config.get_settings_file_path()
    if settings_path.is_file():
        try:
            settings_path.unlink()
        except Exception:
            pass
            
    # Reset in-memory config to env-defined values
    config.INPUT_FOLDERS = [
        Path(p.strip()) for p in os.environ.get("INPUT_FOLDERS", "").split(",") if p.strip()
    ]
    if not config.INPUT_FOLDERS:
        config.INPUT_FOLDERS = [config.PROJECT_ROOT / "media_input"]
    config.OUTPUT_FOLDER = Path(os.environ.get("OUTPUT_FOLDER", str(config.PROJECT_ROOT / "media_output")))
    if not os.environ.get("DB_PATH"):
        config.DB_PATH = config.OUTPUT_FOLDER / "catalog_history.db"
    else:
        config.DB_PATH = Path(os.environ.get("DB_PATH"))
        
    yield
    
    # Cleanup after test
    if settings_path.is_file():
        try:
            settings_path.unlink()
        except Exception:
            pass
            
    # Restore original config
    config.INPUT_FOLDERS = original_input_folders
    config.OUTPUT_FOLDER = original_output_folder
    config.DB_PATH = original_db_path


def test_get_settings():
    client = TestClient(app)
    response = client.get("/api/settings")
    assert response.status_code == 200
    data = response.json()
    
    assert "input_folders" in data
    assert "output_folder" in data
    assert "default_input_folders" in data
    assert "default_output_folder" in data
    assert data["is_custom_input"] is False
    assert data["is_custom_output"] is False


def test_post_settings_update_and_reset():
    client = TestClient(app)
    settings_path = config.get_settings_file_path()
    
    # 1. Update settings with custom folders
    custom_inputs = ["C:\\custom_input_1", "D:\\custom_input_2"]
    custom_output = "C:\\custom_output_dir"
    
    payload = {
        "input_folders": custom_inputs,
        "output_folder": custom_output
    }
    
    response = client.post("/api/settings", json=payload)
    assert response.status_code == 200
    
    # Check JSON response
    data = response.json()
    assert data["status"] == "success"
    assert len(data["input_folders"]) == 2
    assert data["output_folder"] == custom_output
    
    # Verify in-memory config updated
    assert [str(p) for p in config.INPUT_FOLDERS] == custom_inputs
    assert str(config.OUTPUT_FOLDER) == custom_output
    
    # Verify settings.json created and correct
    assert settings_path.is_file()
    with open(settings_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
        assert saved["INPUT_FOLDERS"] == custom_inputs
        assert saved["OUTPUT_FOLDER"] == custom_output
        
    # Verify GET returns custom state
    response_get = client.get("/api/settings")
    data_get = response_get.json()
    assert data_get["is_custom_input"] is True
    assert data_get["is_custom_output"] is True
    
    # 2. Reset settings (POST empty list/string)
    payload_reset = {
        "input_folders": [],
        "output_folder": ""
    }
    response_reset = client.post("/api/settings", json=payload_reset)
    assert response_reset.status_code == 200
    
    # Verify settings.json was deleted
    assert not settings_path.is_file()
    
    # Verify config reset to default
    response_get2 = client.get("/api/settings")
    data_get2 = response_get2.json()
    assert data_get2["is_custom_input"] is False
    assert data_get2["is_custom_output"] is False


def test_select_folder_endpoint(monkeypatch):
    client = TestClient(app)
    
    # Mock tkinter and filedialog.askdirectory to return a custom path without opening a UI
    mock_askdirectory = MagicMock(return_value="C:\\selected_catalog_path")
    
    # We patch inside the select_folder endpoint imports: tkinter.filedialog.askdirectory
    import tkinter.filedialog
    monkeypatch.setattr(tkinter.filedialog, "askdirectory", mock_askdirectory)
    
    # Mock tkinter.Tk to prevent raising TclError on headless systems
    mock_tk = MagicMock()
    monkeypatch.setattr(tkinter, "Tk", mock_tk)
    
    response = client.post("/api/select-folder")
    assert response.status_code == 200
    data = response.json()
    assert data["folder"] == "C:\\selected_catalog_path"
