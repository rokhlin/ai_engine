import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.utils.gemini import PhotoAnalysis, VideoAnalysis, TagItem, analyze_photo
from src import db_sqlite
from src import config
import api

def test_tag_item_from_any():
    # From TagItem
    ti = TagItem(tag="test", category="custom", confidence=0.8)
    assert TagItem.from_any(ti) == ti

    # From flat string
    ti_flat = TagItem.from_any("sunset")
    assert ti_flat.tag == "sunset"
    assert ti_flat.category == "general"
    assert ti_flat.confidence == 1.0

    # From prefixed string
    ti_prefixed = TagItem.from_any("scene:beach")
    assert ti_prefixed.tag == "beach"
    assert ti_prefixed.category == "scene"
    assert ti_prefixed.confidence == 1.0

    # From dict
    ti_dict = TagItem.from_any({"tag": "invoice", "category": "documents", "confidence": 0.95})
    assert ti_dict.tag == "invoice"
    assert ti_dict.category == "documents"
    assert ti_dict.confidence == 0.95

def test_photo_analysis_tags_normalization():
    # Test normalization from mixed list of strings and dicts
    analysis = PhotoAnalysis(
        summary="Test photo",
        summary_ru="Тестовое фото",
        description="A detailed test photo description",
        description_ru="Детальное описание",
        environment="outdoor",
        lighting="natural",
        lighting_ru="естественное",
        time_of_day="day",
        time_of_day_ru="день",
        content_type="nature",
        tags=["nature:mountains", "snow", {"tag": "alps", "category": "location", "confidence": 0.9}]
    )
    assert analysis.content_type == "nature"
    assert len(analysis.tags) == 3
    assert analysis.tags[0].tag == "mountains"
    assert analysis.tags[0].category == "nature"
    assert analysis.tags[1].tag == "snow"
    assert analysis.tags[1].category == "general"
    assert analysis.tags[2].tag == "alps"
    assert analysis.tags[2].category == "location"
    assert analysis.tags[2].confidence == 0.9

def test_video_analysis_tags_normalization():
    analysis = VideoAnalysis(
        summary="Test video",
        summary_ru="Тестовое видео",
        transcription="Hello world",
        transcription_ru="Привет мир",
        timeline_events=[],
        content_type="documents",
        tags=["document", "social:presentation"]
    )
    assert analysis.content_type == "documents"
    assert len(analysis.tags) == 2
    assert analysis.tags[0].tag == "document"
    assert analysis.tags[1].tag == "presentation"
    assert analysis.tags[1].category == "social"

def test_db_save_media_tags_robustness(tmp_path):
    db_file = tmp_path / "test_tags.db"
    old_db = config.DB_PATH
    config.DB_PATH = db_file
    try:
        db_sqlite.init_db()
        media_id = "test_media_123"
        db_sqlite.upsert_media_item(
            media_id=media_id,
            file_path="photos/sample.jpg",
            file_name="sample.jpg",
            media_type="photo",
            file_size=1024,
            mtime=123456.0
        )

        # Mix of string, prefixed string, dict, and TagItem
        tags = [
            "landscape",
            "content_type:nature",
            {"tag": "forest", "category": "flora", "confidence": 0.92},
            TagItem(tag="pine", category="flora", confidence=0.88)
        ]
        db_sqlite.save_media_tags(media_id, tags)

        with db_sqlite.get_db_connection() as conn:
            rows = conn.execute("SELECT tag, category, confidence FROM media_tags WHERE media_id = ? ORDER BY tag", (media_id,)).fetchall()
            tag_names = {r["tag"]: (r["category"], r["confidence"]) for r in rows}
            assert "landscape" in tag_names
            assert tag_names["landscape"][0] == "general"
            assert "nature" in tag_names
            assert tag_names["nature"][0] == "content_type"
            assert "forest" in tag_names
            assert tag_names["forest"][0] == "flora"
            assert "pine" in tag_names
            assert tag_names["pine"][0] == "flora"
    finally:
        config.DB_PATH = old_db

def test_analyze_with_tags_api_endpoint(tmp_path):
    client = TestClient(api.app)
    
    # Create dummy image
    test_img = tmp_path / "doc_test.jpg"
    test_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xFF\xC0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xFF\xC4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xFF\xDA\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xFF\xD9")

    mock_result = {
        "file_path": str(test_img),
        "content_type": "documents",
        "tags": [
            {"tag": "receipt", "category": "documents", "confidence": 0.98},
            {"tag": "finance", "category": "general", "confidence": 0.9}
        ],
        "gemini_analysis": {
            "summary": "Financial document receipt",
            "content_type": "documents",
            "tags": [
                {"tag": "receipt", "category": "documents", "confidence": 0.98},
                {"tag": "finance", "category": "general", "confidence": 0.9}
            ]
        }
    }

    with patch("src.workers.analyze_single_file", return_value=mock_result):
        resp = client.post("/api/ai/analyze-with-tags", json={
            "file": str(test_img),
            "target_tags": ["documents", "receipt", "invoice"],
            "tag_format": "categorized"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["content_type"] == "documents"
        assert len(data["tags"]) == 2
        assert data["tags"][0]["tag"] == "receipt"


def test_format_people_prompt_section_photo():
    from src.utils.gemini import format_people_prompt_section
    
    # 1. Unnamed faces should produce empty section
    unnamed = [{"face_id": "face_1", "name": "face_1", "bbox": [10, 20, 100, 200]}]
    assert format_people_prompt_section(unnamed, is_video=False) == ""

    # 2. Named faces should produce instructions with bounding boxes and forbidden coordinates instruction
    named = [
        {"face_id": "face_1", "name": "Alice", "bbox": [10, 20, 100, 200]},
        {"face_id": "face_2", "name": "Bob", "bbox": [150, 30, 250, 220]}
    ]
    res = format_people_prompt_section(named, is_video=False)
    assert 'Person 1: "Alice" (bounding box: [10, 20, 100, 200])' in res
    assert 'Person 2: "Bob" (bounding box: [150, 30, 250, 220])' in res
    assert "Do NOT mention raw coordinates" in res
    assert "summary" in res
    assert "description" in res


def test_format_people_prompt_section_video():
    from src.utils.gemini import format_people_prompt_section
    
    face_intervals = {
        "face_1": {"name": "Alice", "intervals": ["00:05 - 00:20", "01:10 - 01:30"]},
        "face_2": {"name": "face_2", "intervals": ["00:15 - 00:25"]}
    }
    res = format_people_prompt_section(face_intervals, is_video=True)
    assert 'Person 1: "Alice" (visible during timecodes: 00:05 - 00:20, 01:10 - 01:30)' in res
    assert "timeline_events" in res
    assert "Do NOT output raw tracking IDs" in res


def test_build_vision_prompt():
    from src.utils.gemini import build_vision_prompt

    custom_tmpl = "Analyze {media_type}.\n\nContext:\n{context}\n\nPeople:\n{people}\n\nTags:\n{tag_instructions}"
    prompt = build_vision_prompt(
        media_type="photo",
        context_str="EXIF: Canon EOS",
        people_str="People: Alice",
        tag_instructions="Tags: family",
        custom_template=custom_tmpl
    )
    assert "Analyze photo." in prompt
    assert "Context:\nEXIF: Canon EOS" in prompt
    assert "People:\nPeople: Alice" in prompt
    assert "Tags:\nTags: family" in prompt

    # Test fallback appending when placeholders are omitted
    simple_tmpl = "Simple prompt without variables."
    fallback_prompt = build_vision_prompt(
        media_type="photo",
        context_str="EXIF: Canon EOS",
        people_str="People: Alice",
        tag_instructions="Tags: family",
        custom_template=simple_tmpl
    )
    assert "Simple prompt without variables." in fallback_prompt
    assert "EXIF: Canon EOS" in fallback_prompt
    assert "People: Alice" in fallback_prompt
    assert "Tags: family" in fallback_prompt


def test_vision_prompt_template_api_settings():
    client = TestClient(api.app)
    
    # 1. GET settings returns template and default
    res = client.get("/api/settings")
    assert res.status_code == 200
    data = res.json()
    assert "vision_prompt_template" in data
    assert "default_vision_prompt_template" in data
    assert "{media_type}" in data["default_vision_prompt_template"]

    # 2. POST settings with custom template
    custom_text = "Customized vision prompt: {media_type} with {people}"
    update_res = client.post("/api/settings", json={"vision_prompt_template": custom_text})
    assert update_res.status_code == 200
    
    get_res = client.get("/api/settings")
    assert get_res.json()["vision_prompt_template"] == custom_text

    # 3. Reset template by passing empty string
    reset_res = client.post("/api/settings", json={"vision_prompt_template": ""})
    assert reset_res.status_code == 200
    assert reset_res.json()["vision_prompt_template"] == data["default_vision_prompt_template"]


def test_vision_prompt_template_in_workers_analysis(monkeypatch, tmp_path):
    from src import workers
    from src.utils.gemini import PhotoAnalysis

    captured_kwargs = {}
    def mock_analyze_photo(image_path, **kwargs):
        captured_kwargs.update(kwargs)
        return PhotoAnalysis(
            summary="Test",
            summary_ru="Тест",
            description="Test desc",
            description_ru="Тест описание",
            environment="indoor",
            lighting="natural",
            weather="unknown",
            time_of_day="day",
            content_type="other",
            tags=[]
        )
    
    monkeypatch.setattr("src.utils.gemini.analyze_photo", mock_analyze_photo)
    test_img = tmp_path / "test.jpg"
    test_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")

    custom_tmpl = "Analyze {media_type} with custom rules: {context}"
    workers.process_photo(test_img, custom_prompt=custom_tmpl)
    assert captured_kwargs.get("custom_prompt") == custom_tmpl

    # Also test via analyze_single_file with settings
    captured_kwargs.clear()
    monkeypatch.setattr("src.database.init_db", lambda: None)
    monkeypatch.setattr("src.database.update_sync_record", lambda **kwargs: None)
    monkeypatch.setattr("src.database.get_sync_record", lambda path: None)
    monkeypatch.setattr("src.database.get_all_sync_records", lambda: {})
    workers.analyze_single_file(str(test_img), settings={"vision_prompt_template": custom_tmpl})
    assert captured_kwargs.get("custom_prompt") == custom_tmpl
