import pytest
from src.workers import normalize_modes, get_file_modules_status, ALL_SUPPORTED_MODES

def test_normalize_modes_all():
    assert normalize_modes(None) == ALL_SUPPORTED_MODES
    assert normalize_modes("all") == ALL_SUPPORTED_MODES
    assert normalize_modes(["all"]) == ALL_SUPPORTED_MODES
    assert normalize_modes([]) == ALL_SUPPORTED_MODES

def test_normalize_modes_single_and_multiple():
    assert normalize_modes("faces") == {"faces"}
    assert normalize_modes("faces, duplicates") == {"faces", "duplicates"}
    assert normalize_modes(["faces", "vision"]) == {"faces", "vision"}
    assert normalize_modes("transcribe;faces") == {"transcribe", "faces"}

def test_normalize_modes_ignores_invalid():
    res = normalize_modes(["faces", "invalid_module"])
    assert res == {"faces"}

def test_get_file_modules_status_photo():
    sidecar = {
        "faces": [{"face_id": "face_1"}],
        "phash": "a1b2c3d4e5f6",
        "gemini_analysis": {"description": "A sunny day"}
    }
    status = get_file_modules_status(sidecar, media_type="photo")
    assert status["transcribe"] is None
    assert status["faces"] is True
    assert status["duplicates"] is True
    assert status["vision"] is True

def test_get_file_modules_status_video():
    sidecar = {
        "transcription": "Hello world",
        "face_timeline": {"1.0": ["face_1"]},
        "phash": "123456",
        "gemini_analysis": {"error": "Quota exceeded"}
    }
    status = get_file_modules_status(sidecar, media_type="video")
    assert status["transcribe"] is True
    assert status["faces"] is True
    assert status["duplicates"] is True
    assert status["vision"] is False
