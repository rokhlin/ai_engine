import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from src import config
from src.user_workspace import get_user_root, get_user_catalog_dir, get_user_db_path, validate_path_in_user_workspace
from src.db_sqlite import user_db_context, get_db_connection, init_db
from src import database
from api import app

client = TestClient(app)

@pytest.fixture
def temp_storage():
    tmp_dir = Path(tempfile.mkdtemp(prefix="mt_storage_"))
    old_base = config.USERS_BASE_DIR
    config.USERS_BASE_DIR = tmp_dir
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)
    config.USERS_BASE_DIR = old_base

def test_user_workspace_paths(temp_storage):
    user_id = "usr_12345"
    root = get_user_root(user_id, default_root=str(temp_storage))
    catalog = get_user_catalog_dir(user_id, default_root=str(temp_storage))
    db_path = get_user_db_path(user_id, default_root=str(temp_storage))

    assert str(root).startswith(str(temp_storage))
    assert root.name == user_id
    assert catalog == root / ".catalog"
    assert db_path == catalog / "catalog_history.db"

    # Test path validation
    valid_subpath = root / "photos" / "vacation.jpg"
    assert validate_path_in_user_workspace(user_id, valid_subpath, default_root=str(temp_storage)) is True

    # Test traversal attack
    traversal_path = root / ".." / "other_user" / "secret.jpg"
    assert validate_path_in_user_workspace(user_id, traversal_path, default_root=str(temp_storage)) is False

def test_multi_tenant_db_isolation(temp_storage):
    user_a = "usr_alice"
    user_b = "usr_bob"

    # User A context
    with user_db_context(user_a, default_root=str(temp_storage)):
        init_db()
        database.update_sync_record("/media/photo1.jpg", 100, 1.0, "PROCESSED", "/media/photo1.json")
        rec_a = database.get_sync_record("/media/photo1.jpg")
        assert rec_a is not None
        assert rec_a["file_size"] == 100

    # User B context - should NOT see User A's data
    with user_db_context(user_b, default_root=str(temp_storage)):
        init_db()
        rec_b = database.get_sync_record("/media/photo1.jpg")
        assert rec_b is None

    # User A context again - data should still exist
    with user_db_context(user_a, default_root=str(temp_storage)):
        rec_a2 = database.get_sync_record("/media/photo1.jpg")
        assert rec_a2 is not None

def test_api_path_traversal_prevention(temp_storage):
    user_id = "usr_alice"
    user_root = get_user_root(user_id, default_root=str(temp_storage))
    user_root.mkdir(parents=True, exist_ok=True)

    headers = {
        "X-User-ID": user_id,
        "X-User-Root-Path": str(temp_storage)
    }

    # Allowed browse inside workspace
    resp = client.post("/api/fs/browse", json={"path": str(user_root)}, headers=headers)
    assert resp.status_code == 200

    # Path traversal attack attempt
    traversal_path = str(user_root / ".." / ".." / "system")
    resp_attack = client.post("/api/fs/browse", json={"path": traversal_path}, headers=headers)
    assert resp_attack.status_code == 403
