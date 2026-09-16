import pytest
import sqlite3
import numpy as np
import tempfile
from pathlib import Path
from src import config
from src import database

@pytest.fixture(autouse=True)
def setup_test_db():
    # Create temporary file for test DB
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # Override path in config
    old_db_path = config.DB_PATH
    config.DB_PATH = temp_path
    
    # Initialize DB
    database.init_db()
    
    yield
    
    # Remove temporary DB
    try:
        temp_path.unlink()
    except OSError:
        pass
    config.DB_PATH = old_db_path

def test_sync_history():
    file_path = "C:\\test\\photo.jpg"
    size = 1024
    mtime = 1234567.89
    status = "PROCESSED"
    sidecar = "C:\\test\\photo.jpg.json"
    
    # Write record
    database.update_sync_record(file_path, size, mtime, status, sidecar)
    
    # Read record
    rec = database.get_sync_record(file_path)
    assert rec is not None
    assert rec["file_path"] == file_path
    assert rec["file_size"] == size
    assert rec["mtime"] == mtime
    assert rec["status"] == status
    assert rec["sidecar_path"] == sidecar
    
    # Update record
    database.update_sync_record(file_path, size, mtime + 10.0, "PENDING")
    rec = database.get_sync_record(file_path)
    assert rec["status"] == "PENDING"
    assert rec["mtime"] == mtime + 10.0

def test_face_registry():
    emb = np.random.rand(512).astype(np.float32)
    face_id = "FACE_ID_01"
    name = "Test User"
    
    # Registration
    registered_name = database.register_face(face_id, emb, name)
    assert registered_name == name
    
    # Retrieval
    faces = database.get_all_registered_faces()
    assert len(faces) == 1
    assert faces[0][0] == face_id
    assert faces[0][1] == name
    assert np.allclose(faces[0][2], emb)
    
    # Mapping
    mapping = database.get_face_name_mapping()
    assert mapping[face_id] == name
    
    # Renaming
    database.update_face_name(face_id, "New Name")
    mapping2 = database.get_face_name_mapping()
    assert mapping2[face_id] == "New Name"

def test_generate_next_face_id():
    assert database.generate_next_face_id() == "FACE_ID_01"
    
    emb = np.random.rand(512).astype(np.float32)
    database.register_face("FACE_ID_01", emb)
    assert database.generate_next_face_id() == "FACE_ID_02"

def test_multi_reference_faces_and_known_persons():
    emb1 = np.random.rand(512).astype(np.float32)
    emb2 = np.random.rand(512).astype(np.float32)
    
    # Register Alice photo 1
    database.register_face(
        face_id="FACE_ID_01",
        embedding=emb1,
        name="Alice",
        image_path="facess/FACE_ID_01.jpg",
        confidence=0.95,
        source_file="media/photo1.jpg",
        is_reference=1
    )
    
    # Register Alice photo 2 (different time/look)
    database.register_face(
        face_id="FACE_ID_02",
        embedding=emb2,
        name="Alice",
        image_path="facess/FACE_ID_02.jpg",
        confidence=0.88,
        source_file="media/photo2.jpg",
        is_reference=1
    )
    
    # Get known persons
    persons = database.get_known_persons()
    assert len(persons) == 1
    alice = persons[0]
    assert alice["name"] == "Alice"
    assert alice["reference_count"] == 2
    assert len(alice["reference_faces"]) == 2
    assert {f["face_id"] for f in alice["reference_faces"]} == {"FACE_ID_01", "FACE_ID_02"}
    
    # Check full reference list
    refs = database.get_all_reference_faces_detailed()
    assert len(refs) == 2

def test_unrecognized_faces_and_assignment():
    emb_unrec = np.random.rand(512).astype(np.float32)
    
    # Register unrecognized candidate face
    database.register_face(
        face_id="FACE_ID_10",
        embedding=emb_unrec,
        name="FACE_ID_10",
        image_path="facess/FACE_ID_10.jpg",
        confidence=0.55,
        source_file="media/group.jpg",
        is_reference=0
    )
    
    # Should show up in unrecognized faces
    unrec = database.get_unrecognized_faces()
    assert len(unrec) == 1
    assert unrec[0]["face_id"] == "FACE_ID_10"
    
    # Should NOT show up in active reference faces
    refs = database.get_all_registered_faces(only_references=True)
    assert len(refs) == 0
    
    # Assign unrecognized face to "Bob"
    assigned = database.assign_face_to_person("FACE_ID_10", "Bob")
    assert assigned is True
    
    # Now Bob should be in known persons
    persons = database.get_known_persons()
    assert len(persons) == 1
    assert persons[0]["name"] == "Bob"
    
    # Unrecognized list should now be empty
    assert len(database.get_unrecognized_faces()) == 0

def test_delete_face():
    emb = np.random.rand(512).astype(np.float32)
    database.register_face("FACE_ID_99", emb, "Temp Face")
    assert len(database.get_all_registered_faces()) == 1
    
    deleted = database.delete_face("FACE_ID_99")
    assert deleted is True
    assert len(database.get_all_registered_faces()) == 0

