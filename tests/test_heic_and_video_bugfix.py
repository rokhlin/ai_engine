import io
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from src import config
from src import workers
from src.utils import gemini
from src.utils import local_model
from src.utils import exif
from src.utils import faces
from src.utils import hash as phash_util
from src.utils.gemini import VideoAnalysis, PhotoAnalysis, TimelineEvent


def test_analyze_single_file_accepts_force_keyword(monkeypatch):
    """Verify analyze_single_file accepts force parameter without raising TypeError."""
    called_kwargs = {}

    def mock_process_photo(item, root, **kwargs):
        called_kwargs.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(workers, "process_photo", mock_process_photo)
    monkeypatch.setattr(workers, "find_media_file", lambda f: (Path("photo.jpg"), Path(".")))

    # Should not raise TypeError: analyze_single_file() got an unexpected keyword argument 'force'
    res = workers.analyze_single_file("photo.jpg", force=True)
    assert res == {"status": "ok"}
    assert called_kwargs.get("force") is True


def test_video_analysis_schema_allows_optional_transcription():
    """Verify VideoAnalysis does not fail validation when local model omits transcription fields."""
    partial_json = {
        "summary": "Children playing in park",
        "content_type": "family",
        "tags": [{"tag": "park", "category": "scene", "confidence": 0.9}]
    }

    # Should validate without raising ValidationError for missing transcription_ru or summary_ru
    analysis = VideoAnalysis.model_validate(partial_json)
    assert analysis.summary == "Children playing in park"
    assert analysis.transcription == ""
    assert analysis.transcription_ru == ""
    assert analysis.summary_ru == ""
    assert analysis.timeline_events == []


def test_heic_reading_and_phash_with_pillow_heif(tmp_path):
    """Verify that HEIC images are decoded properly for bytes preparation and pHash."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        heif_available = True
    except ImportError:
        heif_available = False

    if not heif_available:
        pytest.skip("pillow-heif not installed in this environment")

    # Create dummy RGB image and save as HEIC
    img = Image.new("RGB", (100, 100), color=(120, 180, 240))
    heic_path = tmp_path / "sample.heic"
    img.save(heic_path, format="HEIF")

    # Test prepare_image_bytes
    img_bytes = gemini.prepare_image_bytes(heic_path)
    assert isinstance(img_bytes, bytes)
    assert len(img_bytes) > 0

    # Test calculate_phash
    h = phash_util.calculate_phash(heic_path)
    assert h is not None
    assert len(h) > 0


def test_heic_exif_reading(tmp_path):
    """Verify that read_photo_metadata supports getexif for HEIC images."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        heif_available = True
    except ImportError:
        heif_available = False

    if not heif_available:
        pytest.skip("pillow-heif not installed in this environment")

    img = Image.new("RGB", (80, 80), color=(200, 100, 50))
    heic_path = tmp_path / "test_exif.heic"
    img.save(heic_path, format="HEIF")

    meta = exif.read_photo_metadata(heic_path)
    assert isinstance(meta, dict)
    assert "datetime" in meta
    assert "gps" in meta


def test_faces_read_image_bgr_supports_heic(tmp_path):
    """Verify faces module can read HEIC images into OpenCV BGR numpy array."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        heif_available = True
    except ImportError:
        heif_available = False

    if not heif_available:
        pytest.skip("pillow-heif not installed in this environment")

    img = Image.new("RGB", (60, 40), color=(255, 0, 0)) # Red
    heic_path = tmp_path / "red.heic"
    img.save(heic_path, format="HEIF")

    bgr = faces.read_image_bgr(heic_path)
    assert bgr is not None
    assert isinstance(bgr, np.ndarray)
    assert bgr.shape == (40, 60, 3)
    # Red in RGB is (255, 0, 0), in BGR it should be (0, 0, 255) with lossy HEIF tolerance
    assert bgr[20, 30, 2] >= 250
    assert bgr[20, 30, 0] <= 5


def test_analyze_video_local_handles_missing_transcription_from_llm(monkeypatch):
    """Verify analyze_video_local attaches Whisper transcript even if local LLM omits it."""
    # Mock completions to return JSON missing transcription and transcription_ru
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"summary": "A car in the yard", "summary_ru": "Машина во дворе", "timeline_events": [], "tags": []}'
            }
        }]
    }
    monkeypatch.setattr(local_model.requests, "post", mock_post)
    monkeypatch.setattr(local_model, "encode_image_to_base64", lambda path: "dummy_base64")
    monkeypatch.setattr(local_model, "get_loaded_local_models", lambda: ["test-vision-model"])

    # Provide audio transcription externally (as Whisper produces)
    res = local_model.analyze_video_local(
        video_path=Path("dummy.mp4"),
        frames=[Path("frame1.jpg")],
        transcription="Engine sound",
        transcription_ru="Звук двигателя"
    )

    assert isinstance(res, VideoAnalysis)
    assert res.summary == "A car in the yard"
    assert res.transcription == "Engine sound"
    assert res.transcription_ru == "Звук двигателя"
