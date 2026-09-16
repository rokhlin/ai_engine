import pytest
from pathlib import Path
from unittest.mock import MagicMock
import requests
from src import config
from src import workers
from src.utils import gemini
from src.utils import local_model
from src.utils.gemini import PhotoAnalysis, VideoAnalysis, GroupDuplicateAnalysis

def test_model_routing_and_auto_detect(monkeypatch):
    # Reset cached local model name to force auto-detection
    local_model._cached_model_name = None
    old_local_model_name = config.LOCAL_MODEL_NAME
    config.LOCAL_MODEL_NAME = ""

    # Mock requests.get for /models endpoint
    mock_get = MagicMock()
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "data": [{"id": "test-vision-model-loaded"}]
    }
    monkeypatch.setattr(requests, "get", mock_get)

    # Mock requests.post for completions
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"summary": "A test photo", "summary_ru": "Тестовое фото", "description": "A detailed test photo description", "description_ru": "Подробное описание", "environment": "indoor", "lighting": "natural", "lighting_ru": "естественное", "time_of_day": "day", "time_of_day_ru": "день"}'
            }
        }]
    }
    monkeypatch.setattr(requests, "post", mock_post)

    # Mock encode_image_to_base64 to avoid loading real files in PIL
    monkeypatch.setattr(local_model, "encode_image_to_base64", lambda path: "dummy_base64_string")

    # Enable local provider
    old_provider = config.MODEL_PROVIDER
    config.MODEL_PROVIDER = "local"
    
    try:
        # Call analyze_photo
        res = gemini.analyze_photo(Path("tests/test_database.py"))
        assert isinstance(res, PhotoAnalysis)
        assert res.summary == "A test photo"
        assert res.summary_ru == "Тестовое фото"
        
        # Verify requests.get was called to auto-detect
        mock_get.assert_called()
        
        # Verify requests.post was called
        mock_post.assert_called()
    finally:
        config.MODEL_PROVIDER = old_provider
        config.LOCAL_MODEL_NAME = old_local_model_name

def test_quota_fallback(monkeypatch):
    # Mock gemini.analyze_photo to raise a Quota Exceeded exception on first call,
    # and return success on subsequent calls (which will be local model since provider changes).
    call_count = 0
    def mock_analyze_photo(image_path, exif_data=None):
        nonlocal call_count
        call_count += 1
        if config.MODEL_PROVIDER == "gemini":
            raise Exception("429 Resource exhausted: Quota exceeded for model")
        else:
            return PhotoAnalysis(
                summary="Fallback photo",
                summary_ru="Фолбек фото",
                description="Fallback description",
                description_ru="Описание",
                environment="indoor",
                lighting="studio",
                lighting_ru="студийное",
                time_of_day="night",
                time_of_day_ru="ночь"
            )

    monkeypatch.setattr(gemini, "analyze_photo", mock_analyze_photo)

    # Set up config
    old_provider = config.MODEL_PROVIDER
    old_fallback = config.FALLBACK_TO_LOCAL
    config.MODEL_PROVIDER = "gemini"
    config.FALLBACK_TO_LOCAL = True

    try:
        # Call retry_api_call which wraps gemini.analyze_photo
        res = workers.retry_api_call(gemini.analyze_photo, Path("tests/test_database.py"))
        
        # Check that it succeeded on the retry with local model
        assert res.summary == "Fallback photo"
        assert call_count == 2
        assert config.MODEL_PROVIDER == "local"
    finally:
        config.MODEL_PROVIDER = old_provider
        config.FALLBACK_TO_LOCAL = old_fallback

def test_video_local_with_transcription(monkeypatch):
    # Mock requests.post for completions
    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"summary": "A video of children playing", "summary_ru": "Видео с играющими детьми", "transcription": "Hello world", "transcription_ru": "Привет мир", "timeline_events": []}'
            }
        }]
    }
    monkeypatch.setattr(requests, "post", mock_post)

    # Mock encode_image_to_base64
    monkeypatch.setattr(local_model, "encode_image_to_base64", lambda path: "dummy_base64_string")

    # Mock transcribe_video_audio to avoid loading the physical model in unit tests
    from src.utils import transcribe
    monkeypatch.setattr(transcribe, "transcribe_video_audio", lambda path: ("Hello world", "Привет мир"))
    monkeypatch.setattr(local_model, "get_loaded_local_models", lambda: ["test-vision-model"])

    # Enable local provider
    old_provider = config.MODEL_PROVIDER
    config.MODEL_PROVIDER = "local"

    try:
        # Call analyze_video on gemini which routes to local and auto-transcribes
        res = gemini.analyze_video(Path("tests/test_database.py"), frames=[Path("dummy_frame.jpg")])
        assert isinstance(res, VideoAnalysis)
        assert res.summary == "A video of children playing"
        assert res.transcription == "Hello world"
        assert res.transcription_ru == "Привет мир"
    finally:
        config.MODEL_PROVIDER = old_provider

def test_process_video_local_sequence(monkeypatch):
    from src import database
    from src.utils import video, faces, transcribe
    
    # Mock DB functions
    monkeypatch.setattr(database, "update_sync_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(database, "get_face_name_mapping", lambda: {})
    
    # Mock video utilities
    monkeypatch.setattr(video, "read_video_metadata", lambda path: {"duration": 10.0})
    monkeypatch.setattr(video, "compress_video_for_cloud", lambda inp, out: out)
    monkeypatch.setattr(video, "extract_frames_at_1fps", lambda path, out_dir: [Path("dummy_frame.jpg")])
    
    # Mock face detection
    monkeypatch.setattr(faces, "detect_faces", lambda path: [])
    
    # Mock transcribe_video_audio
    transcribe_called = False
    def mock_transcribe(path):
        nonlocal transcribe_called
        transcribe_called = True
        return ("Whisper text", "Текст Whisper")
    monkeypatch.setattr(transcribe, "transcribe_video_audio", mock_transcribe)
    
    # Mock gemini.analyze_video
    analyze_called_with = {}
    def mock_analyze_video(video_path, frames=None, transcription=None, transcription_ru=None):
        nonlocal analyze_called_with
        analyze_called_with = {
            "transcription": transcription,
            "transcription_ru": transcription_ru,
            "transcribe_called_before_analyze": transcribe_called
        }
        return VideoAnalysis(
            summary="A nice video",
            summary_ru="Хорошее видео",
            transcription=transcription or "",
            transcription_ru=transcription_ru or "",
            timeline_events=[]
        )
    monkeypatch.setattr(gemini, "analyze_video", mock_analyze_video)
    
    # Enable local provider
    old_provider = config.MODEL_PROVIDER
    config.MODEL_PROVIDER = "local"
    
    try:
        # We need to pass a file that exists so Path.stat() works
        test_file = Path(__file__).resolve()
        res = workers.process_video(test_file, test_file.parent)
        
        assert res is not None
        assert transcribe_called is True
        assert analyze_called_with["transcription"] == "Whisper text"
        assert analyze_called_with["transcription_ru"] == "Текст Whisper"
        assert analyze_called_with["transcribe_called_before_analyze"] is True
    finally:
        config.MODEL_PROVIDER = old_provider

def test_local_model_authorization_header(monkeypatch):
    local_model._cached_model_name = "test-model"
    old_token = config.LOCAL_API_TOKEN
    config.LOCAL_API_TOKEN = "sk-lm-test-token-123"

    mock_post = MagicMock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"summary": "A test photo", "summary_ru": "Тестовое фото", "description": "Desc", "description_ru": "Опис", "environment": "indoor", "lighting": "natural", "lighting_ru": "ест", "time_of_day": "day", "time_of_day_ru": "день"}'
            }
        }]
    }
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(local_model, "encode_image_to_base64", lambda path: "dummy_base64")
    monkeypatch.setattr(local_model, "get_loaded_local_models", lambda: ["test-model"])

    old_provider = config.MODEL_PROVIDER
    config.MODEL_PROVIDER = "local"

    try:
        gemini.analyze_photo(Path("tests/test_database.py"))
        assert mock_post.called
        headers = mock_post.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer sk-lm-test-token-123"
    finally:
        config.MODEL_PROVIDER = old_provider
        config.LOCAL_API_TOKEN = old_token


def test_get_local_model_name_raises_when_no_models_loaded(monkeypatch):
    local_model._cached_model_name = None
    monkeypatch.setattr(local_model, "get_loaded_local_models", lambda: [])
    
    with pytest.raises(RuntimeError) as exc_info:
        local_model.get_local_model_name()
    assert "No model is currently loaded in LM Studio memory" in str(exc_info.value)


def test_get_local_model_name_reuses_loaded_model(monkeypatch):
    local_model._cached_model_name = None
    monkeypatch.setattr(local_model, "get_loaded_local_models", lambda: ["qwen2.5-vl-7b-instruct"])
    
    model_name = local_model.get_local_model_name()
    assert model_name == "qwen2.5-vl-7b-instruct"


def test_switch_local_model_reuses_already_loaded(monkeypatch):
    monkeypatch.setattr(local_model, "get_loaded_local_models", lambda: ["qwen2.5-vl-7b-instruct"])
    subp_calls = []
    
    def mock_run(cmd, *args, **kwargs):
        subp_calls.append(cmd)
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "ok"
        mock_res.stderr = ""
        return mock_res
        
    monkeypatch.setattr("subprocess.run", mock_run)
    
    res = local_model.switch_local_model("qwen2.5-vl-7b-instruct")
    assert res["success"] is True
    assert "already loaded" in res["message"]
    # Verify no lms load/unload was called
    assert len(subp_calls) == 0


def test_switch_local_model_unloads_all_before_loading_new(monkeypatch):
    monkeypatch.setattr(local_model, "get_loaded_local_models", lambda: ["old-active-model"])
    monkeypatch.setattr("shutil.which", lambda prog: "lms.exe" if prog == "lms" else None)
    
    commands_executed = []
    def mock_run(cmd, *args, **kwargs):
        commands_executed.append(list(cmd))
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = "success"
        mock_res.stderr = ""
        return mock_res
        
    monkeypatch.setattr("subprocess.run", mock_run)
    
    res = local_model.switch_local_model("new-vision-model")
    assert res["success"] is True
    assert len(commands_executed) == 2
    # First command MUST be lms unload -a
    assert commands_executed[0] == ["lms.exe", "unload", "-a"]
    # Second command MUST be lms load new-vision-model -y
    assert commands_executed[1] == ["lms.exe", "load", "new-vision-model", "-y"]



