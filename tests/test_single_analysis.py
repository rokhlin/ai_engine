import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from src import config
from src import workers
from src import database

@pytest.fixture
def setup_temp_folders(monkeypatch):
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        folder1 = temp_dir / "folder1"
        folder2 = temp_dir / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        
        # Create some dummy files
        file1 = folder1 / "sub" / "pic1.jpg"
        file1.parent.mkdir(parents=True, exist_ok=True)
        file1.write_text("fake image data")
        
        file2 = folder2 / "pic2.mp4"
        file2.write_text("fake video data")
        
        # Override config.INPUT_FOLDERS
        monkeypatch.setattr(config, "INPUT_FOLDERS", [folder1, folder2])
        
        yield {
            "folder1": folder1,
            "folder2": folder2,
            "file1": file1,
            "file2": file2
        }

def test_find_media_file_direct_path(setup_temp_folders):
    paths = setup_temp_folders
    file1 = paths["file1"]
    
    # 1. Search by direct path
    res = workers.find_media_file(str(file1))
    assert res is not None
    assert res[0] == file1.resolve()
    assert res[1] == paths["folder1"]

def test_find_media_file_by_name(setup_temp_folders):
    paths = setup_temp_folders
    file1 = paths["file1"]
    file2 = paths["file2"]
    
    # 2. Search by filename only
    res = workers.find_media_file("pic2.mp4")
    assert res is not None
    assert res[0] == file2.resolve()
    assert res[1] == paths["folder2"]
    
    # 3. Search by subpath suffix
    res = workers.find_media_file("sub/pic1.jpg")
    assert res is not None
    assert res[0] == file1.resolve()
    assert res[1] == paths["folder1"]

def test_find_media_file_non_existent(setup_temp_folders):
    res = workers.find_media_file("nonexistent.jpg")
    assert res is None

def test_analyze_single_file_photo(setup_temp_folders, monkeypatch):
    paths = setup_temp_folders
    file1 = paths["file1"]
    
    # Mock database.init_db and faces.init_face_analyzer
    monkeypatch.setattr(database, "init_db", lambda: None)
    
    from src.utils import faces
    monkeypatch.setattr(faces, "init_face_analyzer", lambda: None)
    
    # Mock process_photo
    process_photo_called = False
    def mock_process_photo(photo_path, input_root):
        nonlocal process_photo_called
        process_photo_called = True
        assert photo_path == file1.resolve()
        assert input_root == paths["folder1"]
        return {"media_type": "photo", "file_path": str(photo_path), "exif": {"datetime": "2026:08:21 12:00:00"}, "phash": "0000"}
        
    monkeypatch.setattr(workers, "process_photo", mock_process_photo)
    
    # Mock duplicate groups logic to prevent running real database queries
    monkeypatch.setattr(workers, "process_duplicate_groups", lambda photos: None)
    
    # Mock database connection to return empty list of processed photos
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    monkeypatch.setattr(database, "get_db_connection", lambda: MagicMock(__enter__=lambda s: mock_conn))
    
    workers.analyze_single_file("pic1.jpg")
    assert process_photo_called is True
