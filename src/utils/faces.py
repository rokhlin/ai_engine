import cv2
import numpy as np
import warnings
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

# Suppress scikit-image FutureWarning from internal insightface face alignment calls
warnings.filterwarnings("ignore", category=FutureWarning, message=".*estimate.*")

from src import config
from src import database

# Global flag, lock, and model instance
_INSIGHTFACE_AVAILABLE = False
_face_app = None
_opencv_cascade = None
_face_lock = threading.RLock()

def init_face_analyzer():
    """Initialize face analyzer (InsightFace with CUDA/CPU or OpenCV fallback)."""
    global _INSIGHTFACE_AVAILABLE, _face_app, _opencv_cascade
    
    # Attempt to import InsightFace
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)  # Suppress DRM / device discovery warnings
        
        from insightface.app import FaceAnalysis
        
        # Determine available execution providers, prioritizing CUDA
        available_providers = ort.get_available_providers()
        providers = []
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1
        print(f"Initializing InsightFace with providers: {providers} (ctx_id={ctx_id})...")
        
        # Load buffalo_l model
        app = FaceAnalysis(name='buffalo_l', providers=providers)
        # Initialize with detection size 640x640
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        _face_app = app
        _INSIGHTFACE_AVAILABLE = True
        active_provider = "GPU (CUDA)" if 'CUDAExecutionProvider' in providers else "CPU"
        print(f"InsightFace successfully initialized on {active_provider}.")
        return
    except Exception as e:
        print(f"Warning: Failed to initialize InsightFace: {e}")
        print("Using fallback OpenCV Haar Cascades face detector (embedding comparison will be unavailable).")
        
    # Initialize OpenCV Haar Cascade as fallback
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _opencv_cascade = cv2.CascadeClassifier(cascade_path)
        if _opencv_cascade.empty():
            raise ValueError("Haar Cascade XML not found")
        _INSIGHTFACE_AVAILABLE = False
    except Exception as ex:
        print(f"Critical error: Failed to initialize OpenCV fallback detector: {ex}")

def crop_and_save_face(
    image_source: Union[Path, str, np.ndarray],
    bbox: List[int],
    face_id: str
) -> Optional[str]:
    """Crop the face region with margin and save to OUTPUT_FOLDER/facess/{face_id}.jpg.
    Returns the relative path inside OUTPUT_FOLDER (e.g. 'facess/FACE_ID_01.jpg')."""
    try:
        if isinstance(image_source, (Path, str)):
            img = cv2.imread(str(image_source))
        elif isinstance(image_source, np.ndarray):
            img = image_source
        else:
            return None

        if img is None or img.size == 0 or len(bbox) < 4:
            return None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        
        # Add 15% margin around face crop for better context
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = int(bw * 0.15)
        pad_y = int(bh * 0.15)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if face_crop.size == 0:
            return None

        config.FACES_FOLDER.mkdir(parents=True, exist_ok=True)
        filename = f"{face_id}.jpg"
        save_path = config.FACES_FOLDER / filename
        cv2.imwrite(str(save_path), face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return f"facess/{filename}"
    except Exception as e:
        print(f"Error saving face crop for {face_id}: {e}")
        return None

def detect_faces(image_path: Path) -> List[Dict[str, Any]]:
    """Detect faces on image and return bounding boxes and embeddings."""
    global _INSIGHTFACE_AVAILABLE, _face_app, _opencv_cascade
    
    with _face_lock:
        if _face_app is None and _opencv_cascade is None:
            init_face_analyzer()
            
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Failed to read image {image_path}")
            return []
            
        detected_faces = []
        
        if _INSIGHTFACE_AVAILABLE and _face_app is not None:
            try:
                # InsightFace expects BGR
                faces = _face_app.get(img)
                for idx, face in enumerate(faces):
                    bbox = face.bbox.astype(int).tolist() # [x1, y1, x2, y2]
                    embedding = face.embedding # numpy array (512,)
                    confidence = float(face.det_score)
                    detected_faces.append({
                        "bbox": bbox,
                        "embedding": embedding,
                        "confidence": confidence
                    })
            except Exception as e:
                print(f"InsightFace error processing {image_path.name}: {e}")
                
        elif _opencv_cascade is not None:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = _opencv_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    bbox = [int(x), int(y), int(x + w), int(y + h)]
                    detected_faces.append({
                        "bbox": bbox,
                        "embedding": None,
                        "confidence": 1.0
                    })
            except Exception as e:
                print(f"OpenCV Haar Cascade error processing {image_path.name}: {e}")
                
        return detected_faces

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def cluster_faces(
    face_records: List[Dict[str, Any]],
    similarity_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Cluster a list of face records by cosine similarity of embeddings.
    Returns list of clusters with metadata and grouped face IDs."""
    if similarity_threshold is None:
        similarity_threshold = config.FACE_SIMILARITY_THRESHOLD
        
    if not face_records:
        return []
        
    n = len(face_records)
    # Adjacency list for connected components
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    
    for i in range(n):
        emb_i = face_records[i].get("embedding")
        if emb_i is None:
            continue
        for j in range(i + 1, n):
            emb_j = face_records[j].get("embedding")
            if emb_j is not None:
                sim = cosine_similarity(emb_i, emb_j)
                if sim >= similarity_threshold:
                    adj[i].append(j)
                    adj[j].append(i)
                    
    # Find connected components (BFS/DFS)
    visited = set()
    clusters = []
    
    for i in range(n):
        if i in visited:
            continue
        
        # Traverse cluster component
        comp_indices = []
        queue = [i]
        visited.add(i)
        
        while queue:
            curr = queue.pop(0)
            comp_indices.append(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        cluster_faces_list = [face_records[idx] for idx in comp_indices]
        
        # Pick best representative (has image, highest confidence, earliest created)
        sorted_faces = sorted(
            cluster_faces_list,
            key=lambda x: (
                1 if x.get("image_path") else 0,
                x.get("confidence") if x.get("confidence") is not None else 0.0,
                x.get("created_at") or ""
            ),
            reverse=True
        )
        rep = sorted_faces[0]
        
        face_ids = [f["face_id"] for f in cluster_faces_list]
        source_files = list(dict.fromkeys(
            [f["source_file"] for f in cluster_faces_list if f.get("source_file")]
        ))
        confidences = [f["confidence"] for f in cluster_faces_list if f.get("confidence") is not None]
        avg_conf = float(np.mean(confidences)) if confidences else 1.0
        
        # Return cleaned face entries without raw numpy embeddings for JSON safety
        cleaned_faces = []
        for f in cluster_faces_list:
            cleaned_faces.append({
                "face_id": f["face_id"],
                "name": f.get("name") or f["face_id"],
                "person_id": f.get("person_id"),
                "image_path": f.get("image_path"),
                "confidence": f.get("confidence"),
                "source_file": f.get("source_file"),
                "created_at": f.get("created_at"),
                "is_reference": f.get("is_reference", 0),
                "has_embedding": f.get("embedding") is not None
            })
        
        clusters.append({
            "group_id": rep["face_id"],
            "representative_face_id": rep["face_id"],
            "name": rep.get("name") or rep["face_id"],
            "primary_image": rep.get("image_path"),
            "face_ids": face_ids,
            "count": len(face_ids),
            "faces": cleaned_faces,
            "source_files": source_files,
            "avg_confidence": round(avg_conf, 3)
        })
        
    return clusters

def match_or_register_face(
    embedding: Optional[np.ndarray],
    confidence: float = 1.0,
    image_path: Optional[Union[Path, str, np.ndarray]] = None,
    bbox: Optional[List[int]] = None,
    source_file: Optional[str] = None
) -> str:
    """Match face with database multi-reference samples, existing candidate groups,
    or save face crop to facess/ if unrecognized or low confidence.
    Returns face_id."""
    with _face_lock:
        if embedding is None:
            # Without embeddings, if we have crop info, register as unrecognized candidate
            if image_path is not None and bbox is not None:
                new_id = database.generate_next_face_id()
                crop_path = crop_and_save_face(image_path, bbox, new_id)
                database.register_face(
                    face_id=new_id,
                    embedding=None,
                    name=new_id,
                    image_path=crop_path,
                    confidence=confidence,
                    source_file=str(source_file or image_path or ""),
                    is_reference=0
                )
                return new_id
            return "FACE_ID_UNKNOWN"
            
        # 1. Match against known reference faces (is_reference = 1)
        reference_faces = database.get_all_reference_faces_detailed()
        best_ref_match_id = None
        best_ref_sim = -1.0
        
        for ref in reference_faces:
            ref_emb = ref.get("embedding")
            if ref_emb is not None:
                sim = cosine_similarity(embedding, ref_emb)
                if sim > best_ref_sim:
                    best_ref_sim = sim
                    best_ref_match_id = ref["face_id"]
                    
        if best_ref_sim >= config.FACE_SIMILARITY_THRESHOLD and confidence >= config.FACE_CONFIDENCE_THRESHOLD and best_ref_match_id is not None:
            return best_ref_match_id

        # 2. Match against existing candidate/unrecognized faces (is_reference = 0)
        # Group similar unrecognized faces together under the same face_id/cluster
        unrecognized_faces = database.get_unrecognized_faces_detailed()
        best_unrec_match_id = None
        best_unrec_sim = -1.0
        
        for unrec in unrecognized_faces:
            unrec_emb = unrec.get("embedding")
            if unrec_emb is not None:
                sim = cosine_similarity(embedding, unrec_emb)
                if sim > best_unrec_sim:
                    best_unrec_sim = sim
                    best_unrec_match_id = unrec["face_id"]
                    
        if best_unrec_sim >= config.FACE_SIMILARITY_THRESHOLD and confidence >= config.FACE_CONFIDENCE_THRESHOLD and best_unrec_match_id is not None:
            return best_unrec_match_id

        # 3. Completely new unrecognized face: save face crop to OUTPUT_FOLDER/facess and register
        new_id = database.generate_next_face_id()
        crop_path = None
        if image_path is not None and bbox is not None:
            crop_path = crop_and_save_face(image_path, bbox, new_id)
            
        database.register_face(
            face_id=new_id,
            embedding=embedding,
            name=new_id,
            image_path=crop_path,
            confidence=confidence,
            source_file=str(source_file or image_path or ""),
            is_reference=0
        )
        return new_id



