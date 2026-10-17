import pytest
from src.utils.local_model import _parse_json_safely, _repair_truncated_json
from src.utils import transcribe
from src import config

def test_repair_truncated_repetition_loop():
    truncated_repetitive_text = (
        '{\n'
        '  "translated_text": "Четыре Четыре Четыре Четыре Четыре Четыре Четыре'
    )
    repaired = _repair_truncated_json(truncated_repetitive_text)
    assert repaired.endswith('"}')
    parsed = _parse_json_safely(truncated_repetitive_text)
    assert isinstance(parsed, dict)
    assert "translated_text" in parsed
    assert "Четыре" in parsed["translated_text"]

def test_translate_text_local_same_language():
    text = "Hello world"
    # Same language bypass should return immediately without LLM call
    res = transcribe.translate_text_local(text, "en", "en")
    assert res == text
    res_ru = transcribe.translate_text_local("Привет", "ru", "russian")
    assert res_ru == "Привет"

def test_config_timeout_and_fallback():
    assert getattr(config, "STREAM_DOWNLOAD_TIMEOUT", 0) >= 180
    assert hasattr(config, "FALLBACK_TO_GEMINI")
