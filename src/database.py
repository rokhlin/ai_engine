"""
Unified Database Gateway for Media Cataloger.
Delegates to Option A (Enhanced SQLite with FTS5 + Vector Search + Remote/Network Support)
or Option B (Remote PostgreSQL + pgvector).
"""

import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any
from src import config

logger = logging.getLogger(__name__)


def is_postgres() -> bool:
    """Check if PostgreSQL backend is active."""
    return config.DB_BACKEND == "postgres"


def check_db_connection() -> Dict[str, Any]:
    """Check active database health, latency, and capabilities."""
    if is_postgres():
        try:
            from src import db_postgres
            return db_postgres.check_db_connection()
        except Exception as e:
            return {"status": "error", "backend": "postgres", "error": str(e)}
    
    from src import db_sqlite
    return db_sqlite.check_db_connection()


def get_db_connection():
    """Get active database connection or connection context manager."""
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_db_connection()
    from src import db_sqlite
    return db_sqlite.get_db_connection()


def init_db():
    """Initialize database schema, tables, and full-text / vector indexes."""
    if is_postgres():
        try:
            from src import db_postgres
            db_postgres.init_db()
            return
        except Exception as e:
            logger.warning(f"PostgreSQL init failed ({e}). Falling back to SQLite.")

    from src import db_sqlite
    db_sqlite.init_db()


# --- Media Items & Metadata CRUD ---

def upsert_media_item(*args, **kwargs):
    if is_postgres():
        from src import db_postgres
        return db_postgres.upsert_media_item(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.upsert_media_item(*args, **kwargs)


def upsert_media_metadata(*args, **kwargs):
    if is_postgres():
        from src import db_postgres
        return db_postgres.upsert_media_metadata(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.upsert_media_metadata(*args, **kwargs)


def save_media_tags(*args, **kwargs):
    if is_postgres():
        from src import db_postgres
        return db_postgres.save_media_tags(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.save_media_tags(*args, **kwargs)


def save_media_faces(*args, **kwargs):
    if is_postgres():
        from src import db_postgres
        return db_postgres.save_media_faces(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.save_media_faces(*args, **kwargs)


def save_video_timeline_events(*args, **kwargs):
    if is_postgres():
        from src import db_postgres
        return db_postgres.save_video_timeline_events(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.save_video_timeline_events(*args, **kwargs)


def save_media_embedding(*args, **kwargs):
    if is_postgres():
        from src import db_postgres
        return db_postgres.save_media_embedding(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.save_media_embedding(*args, **kwargs)


def hybrid_search(*args, **kwargs) -> Dict[str, Any]:
    """Execute unified hybrid search across media items, tags, faces, FTS, and vector embeddings."""
    if is_postgres():
        from src import db_postgres
        return db_postgres.hybrid_search(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.hybrid_search(*args, **kwargs)


# --- Sync History & Backward Compatibility ---

def get_sync_record(file_path: str) -> Optional[dict]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_sync_record(file_path)
    from src import db_sqlite
    return db_sqlite.get_sync_record(file_path)


def update_sync_record(
    file_path: str,
    file_size: int,
    mtime: float,
    status: str,
    sidecar_path: Optional[str] = None,
    error_message: Optional[str] = None
):
    if is_postgres():
        from src import db_postgres
        return db_postgres.update_sync_record(file_path, file_size, mtime, status, sidecar_path, error_message)
    from src import db_sqlite
    return db_sqlite.update_sync_record(file_path, file_size, mtime, status, sidecar_path, error_message)


def get_all_sync_records() -> Dict[str, dict]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_all_sync_records()
    from src import db_sqlite
    return db_sqlite.get_all_sync_records()


def get_all_media_metadata() -> Dict[str, Dict[str, Any]]:
    if is_postgres():
        try:
            from src import db_postgres
            return db_postgres.get_all_media_metadata()
        except Exception:
            return {}
    from src import db_sqlite
    return db_sqlite.get_all_media_metadata()


def get_faces_count_by_source_file() -> Dict[str, int]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_faces_count_by_source_file()
    from src import db_sqlite
    return db_sqlite.get_faces_count_by_source_file()


def get_all_faces_by_source_file() -> Dict[str, List[Dict[str, Any]]]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_all_faces_by_source_file()
    from src import db_sqlite
    return db_sqlite.get_all_faces_by_source_file()


def get_faces_by_source_file(source_file: str) -> List[Dict[str, Any]]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_faces_by_source_file(source_file)
    from src import db_sqlite
    return db_sqlite.get_faces_by_source_file(source_file)


def get_all_registered_faces(*args, **kwargs) -> List[Tuple[str, str, np.ndarray]]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_all_registered_faces(*args, **kwargs)
    from src import db_sqlite
    return db_sqlite.get_all_registered_faces(*args, **kwargs)


def generate_next_face_id() -> str:
    from src import db_sqlite
    return db_sqlite.generate_next_face_id()


def register_face(*args, **kwargs) -> str:
    from src import db_sqlite
    return db_sqlite.register_face(*args, **kwargs)


def get_all_reference_faces_detailed() -> List[Dict[str, Any]]:
    from src import db_sqlite
    return db_sqlite.get_all_reference_faces_detailed()


def get_unrecognized_faces_detailed() -> List[Dict[str, Any]]:
    from src import db_sqlite
    return db_sqlite.get_unrecognized_faces_detailed()


def get_face_name_mapping() -> Dict[str, str]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_face_name_mapping()
    from src import db_sqlite
    return db_sqlite.get_face_name_mapping()


def save_face_to_registry(
    face_id: str,
    name: str,
    embedding: Optional[np.ndarray],
    image_path: Optional[str] = None,
    confidence: Optional[float] = None,
    source_file: Optional[str] = None,
    is_reference: bool = True
):
    if is_postgres():
        from src import db_postgres
        return db_postgres.save_face_to_registry(face_id, name, embedding, image_path, confidence, source_file, is_reference)
    from src import db_sqlite
    return db_sqlite.save_face_to_registry(face_id, name, embedding, image_path, confidence, source_file, is_reference)


def update_face_name(face_id: str, new_name: str):
    if is_postgres():
        from src import db_postgres
        return db_postgres.update_face_name(face_id, new_name)
    from src import db_sqlite
    return db_sqlite.update_face_name(face_id, new_name)


def assign_face_to_person(face_id: str, target_name: str) -> bool:
    if is_postgres():
        from src import db_postgres
        return db_postgres.assign_face_to_person(face_id, target_name)
    from src import db_sqlite
    return db_sqlite.assign_face_to_person(face_id, target_name)


def assign_group_to_person(face_ids: List[str], target_name: str) -> bool:
    if is_postgres():
        from src import db_postgres
        return db_postgres.assign_group_to_person(face_ids, target_name)
    from src import db_sqlite
    return db_sqlite.assign_group_to_person(face_ids, target_name)


def reset_face_assignment(face_id: str) -> bool:
    if is_postgres():
        from src import db_postgres
        return db_postgres.reset_face_assignment(face_id)
    from src import db_sqlite
    return db_sqlite.reset_face_assignment(face_id)


def reset_face_assignments_by_filename(filename_query: str) -> Dict[str, Any]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.reset_face_assignments_by_filename(filename_query)
    from src import db_sqlite
    return db_sqlite.reset_face_assignments_by_filename(filename_query)


def delete_face(face_id: str) -> bool:
    if is_postgres():
        from src import db_postgres
        return db_postgres.delete_face(face_id)
    from src import db_sqlite
    return db_sqlite.delete_face(face_id)


def get_unrecognized_faces() -> List[Dict[str, Any]]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_unrecognized_faces()
    from src import db_sqlite
    return db_sqlite.get_unrecognized_faces()


def get_unrecognized_face_groups(similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_unrecognized_face_groups(similarity_threshold)
    from src import db_sqlite
    return db_sqlite.get_unrecognized_face_groups(similarity_threshold)


def get_known_persons() -> List[Dict[str, Any]]:
    if is_postgres():
        from src import db_postgres
        return db_postgres.get_known_persons()
    from src import db_sqlite
    return db_sqlite.get_known_persons()


def add_person_to_media_file(file_path: str, person_name: str) -> Dict[str, Any]:
    if is_postgres():
        from src import db_postgres
        if hasattr(db_postgres, "add_person_to_media_file"):
            return db_postgres.add_person_to_media_file(file_path, person_name)
    from src import db_sqlite
    return db_sqlite.add_person_to_media_file(file_path, person_name)


def remove_face_from_media_file(file_path: str, face_id: str) -> bool:
    if is_postgres():
        from src import db_postgres
        if hasattr(db_postgres, "remove_face_from_media_file"):
            return db_postgres.remove_face_from_media_file(file_path, face_id)
    from src import db_sqlite
    return db_sqlite.remove_face_from_media_file(file_path, face_id)

