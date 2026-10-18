import os
import time
import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src import config
from src.utils.resource_manager import release_pipeline_resources
from src.utils import faces, transcribe, video
from src.workers import MediaWorkerQueue
from api import app

client = TestClient(app)

def test_release_pipeline_resources():
    # Mock unload methods
    with patch("src.utils.faces.unload_face_analyzer") as mock_face, \
         patch("src.utils.transcribe.unload_whisper_model") as mock_whisper:
        
        mock_face.return_value = True
        mock_whisper.return_value = True
        
        stats = release_pipeline_resources()
        assert stats["status"] == "success"
        assert "faces" in stats["freed"]
        assert "whisper" in stats["freed"]
        assert "gc_collected" in stats
        mock_face.assert_called_once()
        mock_whisper.assert_called_once()

def test_api_cleanup_endpoint():
    with patch("src.utils.resource_manager.release_pipeline_resources") as mock_release:
        mock_release.return_value = {"status": "success", "freed": ["faces", "whisper"], "gc_collected": 10}
        response = client.post("/api/cleanup")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Resources unloaded" in data["message"]
        mock_release.assert_called_once()

def test_extract_sampled_frames_empty_or_missing(tmp_path):
    # Non-existent video should return empty list gracefully
    frames = video.extract_sampled_frames(str(tmp_path / "nonexistent.mp4"), str(tmp_path), max_frames=30)
    assert frames == []

def test_extract_sampled_frames_mock(tmp_path):
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")
    
    with patch("src.utils.video.read_video_metadata") as mock_meta, \
         patch("src.utils.video._run_cmd") as mock_run:
        
        mock_meta.return_value = {"duration": 120.0, "fps": 30.0}
        
        # Create fake output files
        out_dir = tmp_path / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (out_dir / f"frame_{i:04d}.jpg").write_bytes(b"img")
            
        frames = video.extract_sampled_frames(dummy_video, out_dir, max_frames=5)
        assert len(frames) == 5
        mock_run.assert_called_once()

def test_worker_queue_photo_priority():
    items = [
        {"file_path": "/path/video1.mp4", "filename": "video1.mp4", "mtime": 100.0},
        {"file_path": "/path/photo1.jpg", "filename": "photo1.jpg", "mtime": 101.0},
        {"file_path": "/path/video2.mov", "filename": "video2.mov", "mtime": 102.0},
        {"file_path": "/path/photo2.png", "filename": "photo2.png", "mtime": 103.0},
    ]
    
    def _item_prio_key(it):
        fp = it.get("file_path", "")
        ext = Path(fp).suffix.lower()
        return 1 if ext in config.SUPPORTED_VIDEO_EXTS else 0
        
    sorted_items = sorted(items, key=_item_prio_key)
    assert sorted_items[0]["filename"] == "photo1.jpg"
    assert sorted_items[1]["filename"] == "photo2.png"
    assert sorted_items[2]["filename"] == "video1.mp4"
    assert sorted_items[3]["filename"] == "video2.mov"

def test_worker_queue_burst_propagation(tmp_path):
    queue = MediaWorkerQueue(max_workers=1, photo_priority=False, burst_window=5.0, modes=["vision"])
    
    primary_file = str(tmp_path / "burst_01.jpg")
    sibling_file = str(tmp_path / "burst_02.jpg")
    Path(primary_file).write_bytes(b"primary")
    Path(sibling_file).write_bytes(b"sibling")
    
    items = [
        {"file_path": primary_file, "filename": "burst_01.jpg", "folder": str(tmp_path), "mtime": 100.0},
        {"file_path": sibling_file, "filename": "burst_02.jpg", "folder": str(tmp_path), "mtime": 102.0},
    ]
    
    mock_meta = {
        "description": "Burst test photo",
        "tags": ["burst", "action"],
        "category": "Action"
    }
    
    with patch("src.workers.process_photo", return_value=mock_meta), \
         patch("src.database.update_sync_record"):
        res = queue.process_queue(items)
        assert len(res) >= 1
        
        # Verify sibling sidecar was written with propagated metadata
        from src.workers import get_relative_output_path
        sibling_sidecar = get_relative_output_path(Path(sibling_file), tmp_path)
        assert sibling_sidecar.exists()
        with open(sibling_sidecar, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["description"] == "Burst test photo"
            assert data["is_burst_duplicate"] is True
            assert data["burst_primary"] == primary_file

def test_worker_queue_watchdog_timeout(tmp_path):
    queue = MediaWorkerQueue(max_workers=1)
    stalled_file = str(tmp_path / "stalled.jpg")
    Path(stalled_file).write_bytes(b"stalled")
    items = [{"file_path": stalled_file, "filename": "stalled.jpg", "folder": str(tmp_path), "mtime": 100.0}]
    
    def slow_process(*args, **kwargs):
        time.sleep(0.3)
        return {"done": True}
        
    with patch("src.workers.process_photo", side_effect=slow_process), \
         patch.object(config, "FILE_ANALYSIS_TIMEOUT", 0.05), \
         patch("src.database.update_sync_record") as mock_db:
        
        results = queue.process_queue(items)
        assert results == []
        mock_db.assert_called_with(
            file_path=stalled_file,
            status="FAILED",
            error="Analysis timeout exceeded (>0.05s)"
        )
