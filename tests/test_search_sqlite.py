import pytest
import numpy as np
import tempfile
from pathlib import Path
from src import config
from src import database
from src import db_sqlite


@pytest.fixture(autouse=True)
def setup_sqlite_test_env():
    # Temporary SQLite DB path
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_path = Path(temp_file.name)
    temp_file.close()

    old_backend = config.DB_BACKEND
    old_db_path = config.DB_PATH

    config.DB_BACKEND = "sqlite"
    config.DB_PATH = temp_path

    # Initialize SQLite database & FTS5
    database.init_db()

    yield

    try:
        temp_path.unlink()
        wal_file = temp_path.with_name(temp_path.name + "-wal")
        shm_file = temp_path.with_name(temp_path.name + "-shm")
        if wal_file.exists():
            wal_file.unlink()
        if shm_file.exists():
            shm_file.unlink()
    except OSError:
        pass

    config.DB_BACKEND = old_backend
    config.DB_PATH = old_db_path


def test_sqlite_health_check():
    """Verify SQLite health check reporting."""
    res = database.check_db_connection()
    assert res["status"] == "healthy"
    assert res["backend"] == "sqlite"
    assert "sqlite_version" in res
    assert "latency_ms" in res
    assert res["fts5_enabled"] is True


def test_multi_person_relational_search():
    """Verify multi-person search query: only media containing all requested persons are returned."""
    # Item 1: Contains Lilia and Yan
    database.upsert_media_item(
        media_id="item_01",
        file_path="/media/family_01.jpg",
        file_name="family_01.jpg",
        media_type="photo",
        file_size=2048,
        media_date="2026-06-01"
    )
    database.save_media_faces("item_01", [
        {"face_id": "F1", "name": "Lilia", "confidence": 0.95},
        {"face_id": "F2", "name": "Yan", "confidence": 0.96}
    ])

    # Item 2: Contains only Lilia
    database.upsert_media_item(
        media_id="item_02",
        file_path="/media/lilia_solo.jpg",
        file_name="lilia_solo.jpg",
        media_type="photo",
        file_size=1024,
        media_date="2026-06-02"
    )
    database.save_media_faces("item_02", [
        {"face_id": "F3", "name": "Lilia", "confidence": 0.98}
    ])

    # Search for both Lilia AND Yan
    res_both = database.hybrid_search(persons=["Lilia", "Yan"])
    assert res_both["total"] == 1
    assert res_both["results"][0]["id"] == "item_01"

    # Search for Lilia only
    res_lilia = database.hybrid_search(persons=["Lilia"])
    assert res_lilia["total"] == 2


def test_tags_and_fts5_search():
    """Verify tag and FTS5 search for queries like 'food' or 'mountains'."""
    database.upsert_media_item(
        media_id="item_food",
        file_path="/media/dinner.jpg",
        file_name="dinner.jpg",
        media_type="photo",
        file_size=4096,
        media_date="2026-05-15"
    )
    database.upsert_media_metadata(
        media_id="item_food",
        summary="Delicious pasta and wine on table",
        description="Italian restaurant food dinner evening",
        environment="indoor"
    )
    database.save_media_tags("item_food", [
        {"tag": "food", "category": "keyword", "confidence": 1.0},
        {"tag": "indoor", "category": "environment", "confidence": 1.0}
    ])

    database.upsert_media_item(
        media_id="item_mountains",
        file_path="/media/hiking.jpg",
        file_name="hiking.jpg",
        media_type="photo",
        file_size=5120,
        media_date="2026-07-20"
    )
    database.upsert_media_metadata(
        media_id="item_mountains",
        summary="Scenic mountain range with snow",
        description="Hiking in Swiss Alps mountains with blue sky",
        environment="outdoor"
    )
    database.save_media_tags("item_mountains", [
        {"tag": "mountains", "category": "keyword", "confidence": 1.0},
        {"tag": "outdoor", "category": "environment", "confidence": 1.0}
    ])

    # 1. Search by tag
    tag_res = database.hybrid_search(tags=["food"])
    assert tag_res["total"] == 1
    assert tag_res["results"][0]["id"] == "item_food"

    # 2. Search by FTS keyword
    fts_res = database.hybrid_search(query_text="Alps hiking")
    assert fts_res["total"] == 1
    assert fts_res["results"][0]["id"] == "item_mountains"


def test_vector_semantic_search_in_sqlite():
    """Verify 768-dim vector embedding semantic similarity ranking in SQLite."""
    text1 = "child is playing with a toy gun in the garden"
    text2 = "cooking dinner pasta in the kitchen"
    text3 = "snowboarder sliding on snowy mountain slope"

    np.random.seed(42)
    base_vec = np.random.randn(768).astype(np.float32)
    base_vec /= np.linalg.norm(base_vec)

    # vec1 is very close to base_vec
    vec1 = base_vec + np.random.randn(768).astype(np.float32) * 0.05
    vec1 /= np.linalg.norm(vec1)

    # vec2 and vec3 are random
    vec2 = np.random.randn(768).astype(np.float32)
    vec2 /= np.linalg.norm(vec2)

    vec3 = np.random.randn(768).astype(np.float32)
    vec3 /= np.linalg.norm(vec3)

    for i, (mid, txt, vec) in enumerate([("id_1", text1, vec1), ("id_2", text2, vec2), ("id_3", text3, vec3)]):
        database.upsert_media_item(
            media_id=mid,
            file_path=f"/media/test_{i}.jpg",
            file_name=f"test_{i}.jpg",
            media_type="photo",
            file_size=1024
        )
        database.upsert_media_metadata(media_id=mid, summary=txt, description=txt)
        database.save_media_embedding(media_id=mid, text_embedding=vec, indexed_text=txt)

    # Search with natural language query close to base_vec / vec1
    query_vec = base_vec.copy()
    search_res = database.hybrid_search(query_vector=query_vec)

    assert search_res["total"] == 3
    # Top result must be item 1 (highest cosine similarity)
    assert search_res["results"][0]["id"] == "id_1"
    assert search_res["results"][0]["search_score"] > search_res["results"][1]["search_score"]
