import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from api import app
from src import config, database

@pytest.fixture(autouse=True)
def setup_dashboard_test_env():
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    
    old_output = config.OUTPUT_FOLDER
    old_faces = config.FACES_FOLDER
    old_db = config.DB_PATH
    
    config.OUTPUT_FOLDER = temp_path
    config.FACES_FOLDER = temp_path / "facess"
    config.DB_PATH = temp_path / "test_dashboard.db"
    
    config.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    config.FACES_FOLDER.mkdir(parents=True, exist_ok=True)
    database.init_db()
    
    yield
    
    config.OUTPUT_FOLDER = old_output
    config.FACES_FOLDER = old_faces
    config.DB_PATH = old_db
    try:
        temp_dir.cleanup()
    except Exception:
        pass


def test_health_check():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "media_cataloger"


def test_dashboard_api_endpoints():
    client = TestClient(app)
    
    # Status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    assert "status" in res_status.json()

    # Settings
    res_settings = client.get("/api/settings")
    assert res_settings.status_code == 200
    assert "input_folders" in res_settings.json()

    # Faces
    res_faces = client.get("/api/faces")
    assert res_faces.status_code == 200
    assert isinstance(res_faces.json(), list)

    # Persons
    res_persons = client.get("/api/faces/persons")
    assert res_persons.status_code == 200
    assert isinstance(res_persons.json(), list)

    # Unrecognized
    res_unrec = client.get("/api/faces/unrecognized")
    assert res_unrec.status_code == 200
    assert isinstance(res_unrec.json(), list)

    # Logs
    res_logs = client.get("/api/logs")
    assert res_logs.status_code == 200
    assert "logs" in res_logs.json()

def test_face_api_crud_workflow():
    client = TestClient(app)
    from src import database
    import numpy as np

    emb = np.random.rand(512).astype(np.float32)
    database.register_face("FACE_ID_API_01", emb, name="FACE_ID_API_01", is_reference=0)

    # Verify in unrecognized
    unrec = client.get("/api/faces/unrecognized").json()
    assert any(f["face_id"] == "FACE_ID_API_01" for f in unrec)

    # Assign face to "David"
    res_assign = client.post("/api/faces/assign", json={"face_id": "FACE_ID_API_01", "name": "David"})
    assert res_assign.status_code == 200

    # Verify now in persons
    persons = client.get("/api/faces/persons").json()
    assert any(p["name"] == "David" for p in persons)

    # Rename David to "David Smith"
    res_rename = client.post("/api/faces/rename", json={"face_id": "FACE_ID_API_01", "name": "David Smith"})
    assert res_rename.status_code == 200
    persons2 = client.get("/api/faces/persons").json()
    assert any(p["name"] == "David Smith" for p in persons2)

    # Delete face
    res_del = client.post("/api/faces/delete", json={"face_id": "FACE_ID_API_01"})
    assert res_del.status_code == 200
    persons3 = client.get("/api/faces/persons").json()
    assert not any(p["name"] == "David Smith" for p in persons3)

def test_face_api_group_and_reset_endpoints():
    client = TestClient(app)
    from src import database
    import numpy as np

    emb1 = np.random.rand(512).astype(np.float32)
    emb2 = np.random.rand(512).astype(np.float32)
    database.register_face("FACE_ID_GRP_01", emb1, name="FACE_ID_GRP_01", is_reference=0, source_file="vacation.jpg")
    database.register_face("FACE_ID_GRP_02", emb2, name="FACE_ID_GRP_02", is_reference=0, source_file="vacation.jpg")

    # 1. Unrecognized groups endpoint
    res_groups = client.get("/api/faces/unrecognized-groups")
    assert res_groups.status_code == 200
    groups = res_groups.json()
    assert isinstance(groups, list)

    # 2. Assign group endpoint
    res_assign_grp = client.post(
        "/api/faces/assign-group",
        json={"face_ids": ["FACE_ID_GRP_01", "FACE_ID_GRP_02"], "name": "Elena"}
    )
    assert res_assign_grp.status_code == 200
    assert "Elena" in res_assign_grp.json()["message"]

    persons = client.get("/api/faces/persons").json()
    assert any(p["name"] == "Elena" for p in persons)

    # 3. Reset face endpoint
    res_reset_face = client.post(
        "/api/faces/reset",
        json={"face_id": "FACE_ID_GRP_01"}
    )
    assert res_reset_face.status_code == 200
    assert res_reset_face.json()["status"] == "success"

    # 4. Reset by filename endpoint
    res_reset_file = client.post(
        "/api/faces/reset-by-filename",
        json={"filename": "vacation.jpg"}
    )
    assert res_reset_file.status_code == 200
    assert res_reset_file.json()["reset_count"] >= 1

def test_media_gallery_endpoints():
    client = TestClient(app)
    from src import database, config
    import tempfile
    from pathlib import Path

    # Create dummy media file in output or temp
    temp_dir = Path(tempfile.gettempdir())
    dummy_img = config.OUTPUT_FOLDER / "test_gallery_sample.jpg"
    dummy_img.write_text("fake image data")

    try:
        # Register face associated with this dummy image
        database.register_face(
            "FACE_ID_MEDIA_TEST_01",
            None,
            name="Sample Person",
            source_file=str(dummy_img),
            is_reference=1
        )

        # 1. Test /api/media/files
        res_files = client.get("/api/media/files")
        assert res_files.status_code == 200
        data = res_files.json()
        assert "total_files" in data
        assert "files" in data
        assert isinstance(data["files"], list)
        if data["files"]:
            first_file = data["files"][0]
            assert "description" in first_file
            assert "description_ru" in first_file

        # 2. Test /api/media/file
        res_file = client.get(f"/api/media/file?path={dummy_img.name}")
        assert res_file.status_code == 200

        # 3. Test /api/media/faces-for-file
        res_faces = client.get(f"/api/media/faces-for-file?file={dummy_img.name}")
        assert res_faces.status_code == 200

        # 4. Test /api/media/add-person
        res_add_person = client.post("/api/media/add-person", json={
            "file": dummy_img.name,
            "name": "Test Person Alice"
        })
        assert res_add_person.status_code == 200
        add_data = res_add_person.json()
        assert add_data["status"] == "success"
        manual_face_id = add_data["data"]["face_id"]
        assert manual_face_id.startswith("manual_")

        # Verify face now appears in faces-for-file
        res_faces_updated = client.get(f"/api/media/faces-for-file?file={dummy_img.name}")
        assert res_faces_updated.status_code == 200
        face_names = [f["name"] for f in res_faces_updated.json()]
        assert "Test Person Alice" in face_names

        # 5. Test /api/media/remove-face
        res_remove = client.post("/api/media/remove-face", json={
            "file": dummy_img.name,
            "face_id": manual_face_id
        })
        assert res_remove.status_code == 200
        assert res_remove.json()["status"] == "success"

        # Verify face is removed from faces-for-file
        res_faces_after = client.get(f"/api/media/faces-for-file?file={dummy_img.name}")
        assert res_faces_after.status_code == 200
        face_names_after = [f["name"] for f in res_faces_after.json()]
        assert "Test Person Alice" not in face_names_after

        # 6. Test /api/media/sidecar with dummy sidecar
        dummy_sidecar = config.OUTPUT_FOLDER / f"{dummy_img.name}.json"
        dummy_sidecar.write_text('{"description": "A sunny day", "description_ru": "Солнечный день", "gemini_analysis": {"lighting": "bright", "lighting_ru": "яркое"}}', encoding="utf-8")
        try:
            res_sidecar = client.get(f"/api/media/sidecar?path={dummy_img.name}")
            assert res_sidecar.status_code == 200
            sidecar_json = res_sidecar.json()
            assert sidecar_json.get("description") == "A sunny day"
            assert sidecar_json.get("description_ru") == "Солнечный день"
        finally:
            if dummy_sidecar.exists():
                dummy_sidecar.unlink()
    finally:
        if dummy_img.exists():
            dummy_img.unlink()
        database.delete_face("FACE_ID_MEDIA_TEST_01")



