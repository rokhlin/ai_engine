import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from src import config
from src import database
from src.utils import faces

@pytest.fixture(autouse=True)
def setup_test_env():
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    
    old_output = config.OUTPUT_FOLDER
    old_faces = config.FACES_FOLDER
    old_db = config.DB_PATH
    
    config.OUTPUT_FOLDER = temp_path
    config.FACES_FOLDER = temp_path / "facess"
    config.DB_PATH = temp_path / "test.db"
    
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

def test_crop_and_save_face():
    # Create synthetic test image 200x200
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = [255, 128, 64]
    
    bbox = [50, 50, 150, 150]
    face_id = "FACE_TEST_CROP"
    
    rel_path = faces.crop_and_save_face(img, bbox, face_id)
    assert rel_path == "facess/FACE_TEST_CROP.jpg"
    
    saved_file = config.FACES_FOLDER / "FACE_TEST_CROP.jpg"
    assert saved_file.is_file()
    
    read_img = cv2.imread(str(saved_file))
    assert read_img is not None
    assert read_img.shape[0] > 0 and read_img.shape[1] > 0

def test_multi_reference_matching():
    # Create base embedding and variations
    base_emb = np.random.rand(512).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)
    
    # Construct similar embedding with known high cosine similarity (~0.95)
    noise = np.random.rand(512).astype(np.float32)
    noise /= np.linalg.norm(noise)
    similar_emb = 0.95 * base_emb + 0.05 * noise
    similar_emb /= np.linalg.norm(similar_emb)
    
    # Very different embedding (orthogonal or independent random)
    diff_emb = np.random.rand(512).astype(np.float32)
    diff_emb /= np.linalg.norm(diff_emb)
    # Ensure diff_emb similarity is below threshold
    if faces.cosine_similarity(base_emb, diff_emb) >= 0.70:
        diff_emb = -base_emb
    
    # Register Alice with base_emb
    database.register_face("FACE_ID_01", base_emb, name="Alice", is_reference=1)
    
    # Matching with similar embedding and high confidence (>0.70)
    matched_id = faces.match_or_register_face(similar_emb, confidence=0.92)
    assert matched_id == "FACE_ID_01"
    
    # Matching with different embedding should fail to match and create a new unrecognized face
    new_face_id = faces.match_or_register_face(diff_emb, confidence=0.85)
    assert new_face_id != "FACE_ID_01"
    
    # Unrecognized faces should contain new_face_id
    unrec = database.get_unrecognized_faces()
    assert any(u["face_id"] == new_face_id for u in unrec)

def test_low_confidence_saves_to_facess():
    # Embedding is identical, but confidence is low (0.50 < 0.70 threshold)
    base_emb = np.random.rand(512).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)
    
    database.register_face("FACE_ID_01", base_emb, name="Alice", is_reference=1)
    
    # Create synthetic image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [20, 20, 80, 80]
    
    # Match with low confidence < 0.70
    face_id = faces.match_or_register_face(
        embedding=base_emb,
        confidence=0.50, # Low confidence (< 70%)
        image_path=img,
        bbox=bbox,
        source_file="test.jpg"
    )
    
    # Should be registered as a new unrecognized face and cropped into facess/
    assert face_id != "FACE_ID_01"
    unrec = database.get_unrecognized_faces()
    assert any(u["face_id"] == face_id for u in unrec)
    
    saved_crop = config.FACES_FOLDER / f"{face_id}.jpg"
    assert saved_crop.is_file()

def test_unrecognized_candidate_matching():
    # When multiple unrecognized faces are processed, similar ones should be grouped under the same face_id
    emb1 = np.random.rand(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    
    noise = np.random.rand(512).astype(np.float32) * 0.05
    emb2 = emb1 + noise
    emb2 /= np.linalg.norm(emb2)
    
    # First unrecognized face
    fid1 = faces.match_or_register_face(emb1, confidence=0.90, source_file="photo1.jpg")
    assert fid1.startswith("FACE_ID_")
    
    # Second face with very similar embedding should match fid1
    fid2 = faces.match_or_register_face(emb2, confidence=0.88, source_file="photo2.jpg")
    assert fid2 == fid1

def test_cluster_faces():
    # Create two distinct clusters of face embeddings
    embA = np.random.rand(512).astype(np.float32)
    embA /= np.linalg.norm(embA)
    
    embA_similar = embA + np.random.rand(512).astype(np.float32) * 0.04
    embA_similar /= np.linalg.norm(embA_similar)
    
    embB = -embA  # Orthogonal/opposite
    embB_similar = embB + np.random.rand(512).astype(np.float32) * 0.04
    embB_similar /= np.linalg.norm(embB_similar)
    
    records = [
        {"face_id": "FACE_A1", "embedding": embA, "confidence": 0.95, "source_file": "img1.jpg", "image_path": "facess/FACE_A1.jpg", "is_reference": 0},
        {"face_id": "FACE_A2", "embedding": embA_similar, "confidence": 0.90, "source_file": "img2.jpg", "image_path": "facess/FACE_A2.jpg", "is_reference": 0},
        {"face_id": "FACE_B1", "embedding": embB, "confidence": 0.92, "source_file": "img3.jpg", "image_path": "facess/FACE_B1.jpg", "is_reference": 0},
        {"face_id": "FACE_B2", "embedding": embB_similar, "confidence": 0.89, "source_file": "img4.jpg", "image_path": "facess/FACE_B2.jpg", "is_reference": 0},
    ]
    
    clusters = faces.cluster_faces(records, similarity_threshold=0.70)
    assert len(clusters) == 2
    
    # Find cluster containing FACE_A1
    cluster_a = next(c for c in clusters if "FACE_A1" in c["face_ids"])
    assert "FACE_A2" in cluster_a["face_ids"]
    assert cluster_a["count"] == 2
    
    cluster_b = next(c for c in clusters if "FACE_B1" in c["face_ids"])
    assert "FACE_B2" in cluster_b["face_ids"]
    assert cluster_b["count"] == 2

def test_assign_group_and_reset():
    # Register candidate faces
    emb1 = np.random.rand(512).astype(np.float32)
    emb2 = np.random.rand(512).astype(np.float32)
    
    database.register_face("FACE_ID_01", emb1, name="FACE_ID_01", is_reference=0, source_file="photo1.jpg")
    database.register_face("FACE_ID_02", emb2, name="FACE_ID_02", is_reference=0, source_file="photo1.jpg")
    
    # Assign group to 'Charlie'
    success = database.assign_group_to_person(["FACE_ID_01", "FACE_ID_02"], "Charlie")
    assert success is True
    
    mapping = database.get_face_name_mapping()
    assert mapping["FACE_ID_01"] == "Charlie"
    assert mapping["FACE_ID_02"] == "Charlie"
    
    ref_faces = database.get_all_reference_faces_detailed()
    assert len(ref_faces) == 2
    
    # Reset single face assignment
    reset_ok = database.reset_face_assignment("FACE_ID_01")
    assert reset_ok is True
    
    mapping_after = database.get_face_name_mapping()
    assert mapping_after["FACE_ID_01"] == "FACE_ID_01"
    assert mapping_after["FACE_ID_02"] == "Charlie"

def test_reset_faces_by_filename():
    emb = np.random.rand(512).astype(np.float32)
    database.register_face("FACE_ID_10", emb, name="David", is_reference=1, source_file="C:/media/event_photo.jpg")
    database.register_face("FACE_ID_11", emb, name="David", is_reference=1, source_file="C:/media/other_photo.jpg")
    
    res = database.reset_face_assignments_by_filename("event_photo.jpg")
    assert res["reset_count"] == 1
    assert "FACE_ID_10" in res["face_ids"]
    
    mapping = database.get_face_name_mapping()
    assert mapping["FACE_ID_10"] == "FACE_ID_10"
    assert mapping["FACE_ID_11"] == "David"

