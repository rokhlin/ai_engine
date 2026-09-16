import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from src import config
from src.workers import (
    MediaWorkerQueue,
    get_queue_status,
    run_pipeline,
    request_pause,
    request_resume,
    request_stop,
    reset_controls,
    is_paused,
    is_stop_requested
)
from src.utils.gemini import GeminiRateLimiter


def test_gemini_rate_limiter():
    """Test thread-safe Gemini rate limiter pacing."""
    limiter = GeminiRateLimiter(rpm_limit=5)
    start_time = time.time()
    
    # Acquire 5 times immediately
    for _ in range(5):
        limiter.acquire()
    
    elapsed_immediate = time.time() - start_time
    assert elapsed_immediate < 1.0, "Initial acquires under limit should be nearly instantaneous"
    assert len(limiter.timestamps) == 5


def test_get_max_workers_config():
    """Test get_max_workers respects active provider settings."""
    original_provider = config.MODEL_PROVIDER
    original_gemini = config.GEMINI_MAX_WORKERS
    original_local = config.LOCAL_MAX_WORKERS

    try:
        config.GEMINI_MAX_WORKERS = 4
        config.LOCAL_MAX_WORKERS = 2

        config.MODEL_PROVIDER = "gemini"
        assert config.get_max_workers() == 4
        assert config.get_max_workers("gemini") == 4
        assert config.get_max_workers("local") == 2

        config.MODEL_PROVIDER = "local"
        assert config.get_max_workers() == 2
        assert config.get_max_workers("gemini") == 4
    finally:
        config.MODEL_PROVIDER = original_provider
        config.GEMINI_MAX_WORKERS = original_gemini
        config.LOCAL_MAX_WORKERS = original_local


def test_media_worker_queue_parallel_execution(tmp_path):
    """Test MediaWorkerQueue processes tasks concurrently across worker threads."""
    queue = MediaWorkerQueue(max_workers=3)

    # Create dummy media items
    items = []
    for i in range(6):
        p = tmp_path / f"test_{i}.jpg"
        p.touch()
        items.append((p, tmp_path, "photo"))

    active_threads = set()
    lock = threading.Lock()

    def mock_process_photo(photo_path, root, *args, **kwargs):
        with lock:
            active_threads.add(threading.current_thread().name)
        time.sleep(0.05)
        return {"file_path": str(photo_path), "media_type": "photo"}

    with patch("src.workers.process_photo", side_effect=mock_process_photo):
        res = queue.process_queue(items)

    assert len(res) == 6
    assert queue.completed_items == 6
    assert queue.failed_items == 0
    # Verified multiple worker threads were utilized
    assert len(active_threads) >= 2


def test_media_worker_queue_status_reporting(tmp_path):
    """Test get_status reflects accurate worker queue metrics."""
    queue = MediaWorkerQueue(max_workers=4)
    status = queue.get_status()
    
    assert status["max_workers"] == 4
    assert status["active_workers"] == 0
    assert status["completed"] == 0
    assert status["total"] == 0


def test_worker_queue_stop_signal(tmp_path):
    """Test worker queue gracefully terminates when stop is requested."""
    reset_controls()
    queue = MediaWorkerQueue(max_workers=2)

    items = []
    for i in range(10):
        p = tmp_path / f"item_{i}.jpg"
        p.touch()
        items.append((p, tmp_path, "photo"))

    def mock_process_photo(photo_path, root, *args, **kwargs):
        time.sleep(0.05)
        # Request stop after first item finishes
        request_stop()
        return {"file_path": str(photo_path), "media_type": "photo"}

    with patch("src.workers.process_photo", side_effect=mock_process_photo):
        res = queue.process_queue(items)

    assert is_stop_requested()
    assert len(res) < 10, "Queue should stop processing remaining items once stop is signaled"
    reset_controls()


def test_process_photo_ai_failure_marks_failed(tmp_path):
    """Test that an AI analysis failure sets status=FAILED in sync history and does not fake PROCESSED status."""
    from src import database, workers
    from src.utils import exif, hash as img_hash, faces, gemini

    dummy_img = tmp_path / "photo_fail.jpg"
    dummy_img.write_bytes(b"dummy image content")

    sync_calls = []
    def mock_update_sync_record(**kwargs):
        sync_calls.append(kwargs)

    upsert_calls = []
    def mock_upsert_media_item(**kwargs):
        upsert_calls.append(kwargs)

    with patch.object(exif, "read_photo_metadata", return_value={"datetime": "2026-01-01 12:00:00"}), \
         patch.object(faces, "detect_faces", return_value=[]), \
         patch.object(img_hash, "calculate_phash", return_value="0000000000000000"), \
         patch.object(gemini, "analyze_photo", side_effect=RuntimeError("AI semantic model unavailable (400 Bad Request)")), \
         patch.object(database, "update_sync_record", side_effect=mock_update_sync_record), \
         patch.object(database, "upsert_media_item", side_effect=mock_upsert_media_item):

        result = workers.process_photo(dummy_img, tmp_path)

        assert result is None
        # Verify sync record has status FAILED with error_message
        assert any(c.get("status") == "FAILED" and "400 Bad Request" in c.get("error_message", "") for c in sync_calls)
        # Ensure it was never marked as PROCESSED
        assert not any(c.get("status") == "PROCESSED" for c in sync_calls)
        assert not any(c.get("status") == "PROCESSED" for c in upsert_calls)


def test_process_video_ai_failure_marks_failed(tmp_path):
    """Test that a video AI analysis failure sets status=FAILED in sync history."""
    from src import database, workers
    from src.utils import video, faces, gemini

    dummy_vid = tmp_path / "video_fail.mp4"
    dummy_vid.write_bytes(b"dummy video content")

    sync_calls = []
    def mock_update_sync_record(**kwargs):
        sync_calls.append(kwargs)

    upsert_calls = []
    def mock_upsert_media_item(**kwargs):
        upsert_calls.append(kwargs)

    with patch.object(video, "read_video_metadata", return_value={"duration": 10.0}), \
         patch.object(video, "compress_video_for_cloud", return_value=dummy_vid), \
         patch.object(video, "extract_frames_at_1fps", return_value=[]), \
         patch.object(faces, "detect_faces", return_value=[]), \
         patch.object(gemini, "analyze_video", side_effect=RuntimeError("Video AI analysis failed: 500 Internal Error")), \
         patch.object(database, "update_sync_record", side_effect=mock_update_sync_record), \
         patch.object(database, "upsert_media_item", side_effect=mock_upsert_media_item):

        result = workers.process_video(dummy_vid, tmp_path)

        assert result is None
        assert any(c.get("status") == "FAILED" and "500 Internal Error" in c.get("error_message", "") for c in sync_calls)
        assert not any(c.get("status") == "PROCESSED" for c in sync_calls)
        assert not any(c.get("status") == "PROCESSED" for c in upsert_calls)
