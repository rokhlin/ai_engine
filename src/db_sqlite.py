"""
Enhanced SQLite Database Engine & Remote/Network Connector for Media Cataloger.
Provides resilient connection management for local & network shared databases (WAL mode, busy timeout),
full-text search (FTS5 with BM25 ranking), vector similarity search, unified relational schema,
and hybrid multi-modal search.
"""

import sqlite3
import numpy as np
import json
import re
import time
import logging
import threading
import contextvars
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from contextlib import closing, contextmanager

from src import config

logger = logging.getLogger(__name__)
_db_write_lock = threading.RLock()

# Multi-tenant user context variables
_current_user_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_current_user_id", default=None)
_current_user_root: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_current_user_root", default=None)
_current_db_path: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar("_current_db_path", default=None)

def set_current_user_context(user_id: Optional[str], default_root: Optional[str] = None):
    """Set the active user context for the current async task/thread."""
    _current_user_id.set(user_id)
    _current_user_root.set(default_root)
    if user_id:
        from src.user_workspace import get_user_db_path
        _current_db_path.set(get_user_db_path(user_id, default_root))
    else:
        _current_db_path.set(None)

@contextmanager
def user_db_context(user_id: Optional[str], default_root: Optional[str] = None):
    """Context manager to run a block within an isolated user DB context."""
    token_uid = _current_user_id.set(user_id)
    token_root = _current_user_root.set(default_root)
    if user_id:
        from src.user_workspace import get_user_db_path
        token_db = _current_db_path.set(get_user_db_path(user_id, default_root))
    else:
        token_db = _current_db_path.set(None)
    try:
        yield
    finally:
        _current_user_id.reset(token_uid)
        _current_user_root.reset(token_root)
        _current_db_path.reset(token_db)

def get_current_db_path() -> Path:
    """Return the active SQLite database path based on user context or system config."""
    if _current_db_path.get() is not None:
        return _current_db_path.get()
    uid = _current_user_id.get()
    if uid:
        from src.user_workspace import get_user_db_path
        return get_user_db_path(uid, _current_user_root.get())
    return Path(config.DB_PATH)

def get_db_connection(
    db_path: Optional[Union[str, Path]] = None,
    user_id: Optional[str] = None,
    default_root: Optional[str] = None
) -> sqlite3.Connection:
    """
    Establish optimized SQLite connection with network-safe locking,
    WAL journal mode, busy timeout, and memory cache pragmas.
    Supports multi-tenant per-user database paths.
    """
    if db_path is not None:
        target_path = Path(db_path)
    elif user_id is not None:
        from src.user_workspace import get_user_db_path
        target_path = get_user_db_path(user_id, default_root)
    else:
        target_path = get_current_db_path()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.is_file() or target_path.stat().st_size == 0:
        with _db_write_lock:
            from src.migrations import run_migrations
            run_migrations(target_path, auto_backup=False)

    timeout_sec = float(config.SQLITE_BUSY_TIMEOUT_MS) / 1000.0
    conn = sqlite3.connect(str(target_path), timeout=timeout_sec)
    conn.row_factory = sqlite3.Row
    
    # Configure resilient pragmas
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA busy_timeout = {config.SQLITE_BUSY_TIMEOUT_MS};")
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA cache_size = -64000;")  # 64MB memory cache
    
    if config.SQLITE_WAL_MODE:
        try:
            cursor.execute("PRAGMA journal_mode = WAL;")
        except Exception:
            pass
    cursor.close()
    
    return conn


def check_db_connection() -> Dict[str, Any]:
    """Check database connectivity, latency, size, journal mode, and FTS5 status."""
    start = time.perf_counter()
    try:
        with closing(get_db_connection()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT sqlite_version();")
            ver = cur.fetchone()[0]
            
            cur.execute("PRAGMA journal_mode;")
            journal_mode = cur.fetchone()[0]
            
            # Check FTS5
            fts5_ok = False
            try:
                cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(col);")
                cur.execute("DROP TABLE IF EXISTS _fts5_test;")
                fts5_ok = True
            except Exception:
                fts5_ok = False
                
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            db_file = get_current_db_path()
            size_mb = round(db_file.stat().st_size / (1024 * 1024), 2) if db_file.is_file() else 0.0
            
            return {
                "status": "healthy",
                "backend": "sqlite",
                "latency_ms": latency_ms,
                "sqlite_version": ver,
                "journal_mode": journal_mode,
                "fts5_enabled": fts5_ok,
                "db_path": str(db_file),
                "db_size_mb": size_mb,
                "busy_timeout_ms": config.SQLITE_BUSY_TIMEOUT_MS
            }
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {
            "status": "error",
            "backend": "sqlite",
            "latency_ms": latency_ms,
            "error": str(e),
            "db_path": str(get_current_db_path())
        }


def init_db(
    db_path: Optional[Union[str, Path]] = None,
    user_id: Optional[str] = None,
    default_root: Optional[str] = None
):
    """Initialize SQLite database tables, FTS5 virtual table, and indexes via migration runner."""
    from src.migrations import run_migrations
    if db_path is not None:
        target_path = Path(db_path)
    elif user_id is not None:
        from src.user_workspace import get_user_db_path
        target_path = get_user_db_path(user_id, default_root)
    else:
        target_path = get_current_db_path()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with _db_write_lock:
        run_migrations(target_path, auto_backup=(target_path.is_file() and target_path.stat().st_size > 0))



# --- Media & Metadata Ingestion Functions ---

def upsert_media_item(
    media_id: str,
    file_path: str,
    file_name: str,
    media_type: str,
    file_size: int,
    mtime: Optional[float] = None,
    duration: Optional[float] = None,
    media_date: Optional[str] = None,
    phash: Optional[str] = None,
    status: str = "PROCESSED"
) -> str:
    """Insert or update a media item record in SQLite."""
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("""
                    INSERT INTO media_items (
                        id, file_path, file_name, media_type, file_size,
                        mtime, duration, media_date, phash, status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'), datetime('now', 'localtime'))
                    ON CONFLICT(id) DO UPDATE SET
                        file_path = excluded.file_path,
                        file_name = excluded.file_name,
                        media_type = excluded.media_type,
                        file_size = excluded.file_size,
                        mtime = excluded.mtime,
                        duration = excluded.duration,
                        media_date = excluded.media_date,
                        phash = excluded.phash,
                        status = excluded.status,
                        updated_at = datetime('now', 'localtime');
                """, (media_id, file_path, file_name, media_type, file_size, mtime, duration, media_date, phash, status))
        return media_id


def upsert_media_metadata(
    media_id: str,
    summary: Optional[str] = None,
    summary_ru: Optional[str] = None,
    description: Optional[str] = None,
    description_ru: Optional[str] = None,
    environment: Optional[str] = None,
    lighting: Optional[str] = None,
    weather: Optional[str] = None,
    time_of_day: Optional[str] = None,
    ocr_text: Optional[str] = None,
    camera_make: Optional[str] = None,
    camera_model: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None,
    raw_exif: Optional[dict] = None,
    raw_gemini: Optional[dict] = None,
    raw_defects: Optional[dict] = None
):
    """Insert or update structured metadata for a media item."""
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("""
                    INSERT INTO media_metadata (
                        media_id, summary, summary_ru, description, description_ru,
                        environment, lighting, weather, time_of_day, ocr_text,
                        camera_make, camera_model, latitude, longitude, location_name,
                        raw_exif, raw_gemini, raw_defects
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(media_id) DO UPDATE SET
                        summary = excluded.summary,
                        summary_ru = excluded.summary_ru,
                        description = excluded.description,
                        description_ru = excluded.description_ru,
                        environment = excluded.environment,
                        lighting = excluded.lighting,
                        weather = excluded.weather,
                        time_of_day = excluded.time_of_day,
                        ocr_text = excluded.ocr_text,
                        camera_make = excluded.camera_make,
                        camera_model = excluded.camera_model,
                        latitude = excluded.latitude,
                        longitude = excluded.longitude,
                        location_name = excluded.location_name,
                        raw_exif = excluded.raw_exif,
                        raw_gemini = excluded.raw_gemini,
                        raw_defects = excluded.raw_defects;
                """, (
                    media_id, summary, summary_ru, description, description_ru,
                    environment, lighting, weather, time_of_day, ocr_text,
                    camera_make, camera_model, latitude, longitude, location_name,
                    json.dumps(raw_exif) if raw_exif else None,
                    json.dumps(raw_gemini) if raw_gemini else None,
                    json.dumps(raw_defects) if raw_defects else None
                ))
                _refresh_fts_entry(conn, media_id)


def save_media_tags(media_id: str, tags: List[Any]):
    """Save extracted tags for a media item."""
    if not tags:
        return
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("DELETE FROM media_tags WHERE media_id = ?;", (media_id,))
                for item in tags:
                    if isinstance(item, str):
                        item_str = item.strip().lower()
                        if not item_str:
                            continue
                        if ":" in item_str:
                            cat, tag = item_str.split(":", 1)
                            cat = cat.strip()
                            tag = tag.strip()
                        else:
                            cat = "general"
                            tag = item_str
                        conf = 1.0
                    elif isinstance(item, dict):
                        tag = str(item.get("tag") or item.get("name") or "").strip().lower()
                        if not tag:
                            continue
                        cat = str(item.get("category", "general")).strip().lower()
                        try:
                            conf = float(item.get("confidence", 1.0))
                        except (ValueError, TypeError):
                            conf = 1.0
                    elif hasattr(item, "tag"):
                        tag = str(getattr(item, "tag", "")).strip().lower()
                        if not tag:
                            continue
                        cat = str(getattr(item, "category", "general")).strip().lower()
                        try:
                            conf = float(getattr(item, "confidence", 1.0))
                        except (ValueError, TypeError):
                            conf = 1.0
                    else:
                        continue

                    conn.execute("""
                        INSERT INTO media_tags (media_id, tag, category, confidence)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(media_id, tag, category) DO NOTHING;
                    """, (media_id, tag, cat, conf))
                _refresh_fts_entry(conn, media_id)


def save_media_faces(media_id: str, faces: List[Dict[str, Any]]):
    """Save detected and identified faces for a media item."""
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("DELETE FROM media_faces WHERE media_id = ?;", (media_id,))
                for f in faces:
                    name = f.get("name", "Unknown").strip()
                    person_id = f.get("person_id")
                    
                    # Ensure person exists in persons table
                    if name and not name.startswith("face_") and name.lower() != "unknown":
                        if not person_id:
                            person_id = name
                        conn.execute("""
                            INSERT INTO persons (id, name)
                            VALUES (?, ?)
                            ON CONFLICT(name) DO UPDATE SET id = excluded.id;
                        """, (person_id, name))
                    
                    conn.execute("""
                        INSERT INTO media_faces (
                            media_id, face_id, person_id, name, confidence, bbox,
                            image_path, time_start, time_end, time_intervals, is_reference
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        media_id,
                        f.get("face_id", ""),
                        person_id,
                        name,
                        f.get("confidence", 1.0),
                        json.dumps(f.get("bbox")) if f.get("bbox") else None,
                        f.get("image_path"),
                        f.get("time_start"),
                        f.get("time_end"),
                        json.dumps(f.get("time_intervals")) if f.get("time_intervals") else None,
                        1 if f.get("is_reference", True) else 0
                    ))
                _refresh_fts_entry(conn, media_id)


def save_video_timeline_events(media_id: str, events: List[Dict[str, Any]]):
    """Save segmented video timeline events."""
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("DELETE FROM video_timeline_events WHERE media_id = ?;", (media_id,))
                for ev in events:
                    t_start = ev.get("timestamp_start", "00:00")
                    t_end = ev.get("timestamp_end", "00:00")
                    
                    def parse_sec(t_str: str) -> float:
                        parts = str(t_str).split(":")
                        if len(parts) == 2:
                            return int(parts[0]) * 60 + int(parts[1])
                        return 0.0

                    conn.execute("""
                        INSERT INTO video_timeline_events (
                            media_id, timestamp_start, timestamp_end, start_sec, end_sec, activity, activity_ru
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?);
                    """, (
                        media_id,
                        t_start,
                        t_end,
                        parse_sec(t_start),
                        parse_sec(t_end),
                        ev.get("activity", ""),
                        ev.get("activity_ru", "")
                    ))


def save_media_embedding(media_id: str, text_embedding: List[float], indexed_text: str):
    """Save 768-dim float32 binary BLOB vector embedding."""
    arr = np.array(text_embedding, dtype=np.float32)
    blob = arr.tobytes()
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("""
                    INSERT INTO media_embeddings (media_id, text_embedding, indexed_text, created_at)
                    VALUES (?, ?, ?, datetime('now', 'localtime'))
                    ON CONFLICT(media_id) DO UPDATE SET
                        text_embedding = excluded.text_embedding,
                        indexed_text = excluded.indexed_text,
                        created_at = datetime('now', 'localtime');
                """, (media_id, blob, indexed_text))


def _refresh_fts_entry(conn: sqlite3.Connection, media_id: str):
    """Update FTS5 virtual table entry for a media item."""
    try:
        # Fetch metadata
        meta = conn.execute("""
            SELECT m.file_name, meta.summary, meta.description, meta.ocr_text, meta.location_name
            FROM media_items m
            LEFT JOIN media_metadata meta ON m.id = meta.media_id
            WHERE m.id = ?
        """, (media_id,)).fetchone()
        
        if not meta:
            return
            
        # Fetch tags
        tags_rows = conn.execute("SELECT tag FROM media_tags WHERE media_id = ?", (media_id,)).fetchall()
        tags_str = " ".join([r["tag"] for r in tags_rows])
        
        # Fetch faces
        faces_rows = conn.execute("SELECT DISTINCT name FROM media_faces WHERE media_id = ?", (media_id,)).fetchall()
        faces_str = " ".join([r["name"] for r in faces_rows if not r["name"].startswith("face_")])
        
        # Upsert FTS5 table
        conn.execute("DELETE FROM media_fts WHERE media_id = ?;", (media_id,))
        conn.execute("""
            INSERT INTO media_fts (media_id, file_name, summary, description, ocr_text, tags, faces, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            media_id,
            meta["file_name"] or "",
            meta["summary"] or "",
            meta["description"] or "",
            meta["ocr_text"] or "",
            tags_str,
            faces_str,
            meta["location_name"] or ""
        ))
    except Exception as e:
        logger.debug(f"FTS refresh notice: {e}")


# --- Hybrid Search Query Engine for SQLite ---

def hybrid_search(
    query_vector: Optional[List[float]] = None,
    query_text: Optional[str] = None,
    persons: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    media_type: Optional[str] = None,
    camera_model: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    Unified Hybrid Search Engine on SQLite:
    - In-memory Vector Cosine Similarity Search (NumPy float32 dot product)
    - FTS5 BM25 Full-Text Search across summaries, descriptions, OCR, tags, faces
    - Multi-person Relational Filtering (HAVING COUNT = len(persons))
    - Explicit Tag Filtering
    - EXIF, Camera, and Date Filters
    """
    with closing(get_db_connection()) as conn:
        conditions = ["m.status = 'PROCESSED'"]
        params: List[Any] = []

        # 1. Media type filter
        if media_type and media_type.lower() in ("photo", "video"):
            conditions.append("m.media_type = ?")
            params.append(media_type.lower())

        # 2. Camera filter
        if camera_model:
            conditions.append("(meta.camera_model LIKE ? OR meta.camera_make LIKE ?)")
            params.extend([f"%{camera_model}%", f"%{camera_model}%"])

        # 3. Date range filters
        if date_from:
            conditions.append("m.media_date >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("m.media_date <= ?")
            params.append(date_to)

        # 4. Multi-person filter (e.g. "Lilia and Yan in photo")
        person_join = ""
        if persons and len(persons) > 0:
            cleaned_persons = [p.strip().lower() for p in persons if p.strip()]
            if cleaned_persons:
                placeholders = ",".join("?" for _ in cleaned_persons)
                person_join = f"""
                    JOIN (
                        SELECT mf.media_id
                        FROM media_faces mf
                        WHERE LOWER(mf.name) IN ({placeholders})
                        GROUP BY mf.media_id
                        HAVING COUNT(DISTINCT LOWER(mf.name)) >= {len(cleaned_persons)}
                    ) matched_persons ON m.id = matched_persons.media_id
                """
                params.extend(cleaned_persons)

        # 5. Tag filter (e.g. "mountains", "food")
        tag_join = ""
        if tags and len(tags) > 0:
            cleaned_tags = [t.strip().lower() for t in tags if t.strip()]
            if cleaned_tags:
                placeholders = ",".join("?" for _ in cleaned_tags)
                tag_join = f"""
                    JOIN (
                        SELECT mt.media_id
                        FROM media_tags mt
                        WHERE LOWER(mt.tag) IN ({placeholders})
                        GROUP BY mt.media_id
                    ) matched_tags ON m.id = matched_tags.media_id
                """
                params.extend(cleaned_tags)

        # 6. FTS5 Text Search Join if keyword query provided
        fts_join = ""
        fts_score_col = "1.0 AS fts_score"
        if query_text and query_text.strip() and query_vector is None:
            # Clean words for FTS5 match query
            words = [w for w in re.findall(r"\w+", query_text.strip()) if len(w) > 1]
            if words:
                fts_query = " OR ".join(f'"{w}"*' for w in words)
                fts_join = "JOIN (SELECT media_id, bm25(media_fts) as rank FROM media_fts WHERE media_fts MATCH ?) fts_match ON m.id = fts_match.media_id"
                params.insert(0, fts_query)
                fts_score_col = "(1.0 / (1.0 + ABS(fts_match.rank))) AS fts_score"

        where_clause = " AND ".join(conditions)

        # Main candidate retrieval query
        sql = f"""
            SELECT 
                m.id,
                m.file_path,
                m.file_name,
                m.media_type,
                m.file_size,
                m.duration,
                m.media_date,
                m.phash,
                meta.summary,
                meta.summary_ru,
                meta.description,
                meta.description_ru,
                meta.environment,
                meta.lighting,
                meta.weather,
                meta.time_of_day,
                meta.ocr_text,
                meta.camera_make,
                meta.camera_model,
                meta.location_name,
                {fts_score_col},
                emb.text_embedding
            FROM media_items m
            LEFT JOIN media_metadata meta ON m.id = meta.media_id
            LEFT JOIN media_embeddings emb ON m.id = emb.media_id
            {person_join}
            {tag_join}
            {fts_join}
            WHERE {where_clause}
            ORDER BY m.media_date DESC, m.created_at DESC;
        """

        rows = conn.execute(sql, params).fetchall()
        
        # 7. Compute Vector Cosine Similarity if query vector provided
        scored_items = []
        if query_vector is not None and len(query_vector) == 768:
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            for r in rows:
                item = dict(r)
                blob = item.pop("text_embedding", None)
                sim_score = 0.0
                if blob:
                    emb = np.frombuffer(blob, dtype=np.float32)
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm > 0:
                        sim_score = float(np.dot(emb / emb_norm, q_vec))
                
                item["search_score"] = round(sim_score, 4)
                scored_items.append(item)
            
            # Sort by cosine similarity descending
            scored_items.sort(key=lambda x: x["search_score"], reverse=True)
        else:
            for r in rows:
                item = dict(r)
                item.pop("text_embedding", None)
                item["search_score"] = round(item.get("fts_score", 1.0), 4)
                scored_items.append(item)

        total_count = len(scored_items)
        paged_items = scored_items[offset : offset + limit]

        # 8. Enrich results with faces and tags
        if paged_items:
            media_ids = [p["id"] for p in paged_items]
            placeholders = ",".join("?" for _ in media_ids)
            
            # Fetch faces
            faces_rows = conn.execute(f"""
                SELECT media_id, face_id, person_id, name, confidence, bbox, time_start, time_end
                FROM media_faces
                WHERE media_id IN ({placeholders});
            """, media_ids).fetchall()
            faces_map: Dict[str, List[dict]] = {}
            for f in faces_rows:
                mid = f["media_id"]
                if mid not in faces_map:
                    faces_map[mid] = []
                faces_map[mid].append(dict(f))

            # Fetch tags
            tags_rows = conn.execute(f"""
                SELECT media_id, tag, category, confidence
                FROM media_tags
                WHERE media_id IN ({placeholders});
            """, media_ids).fetchall()
            tags_map: Dict[str, List[dict]] = {}
            for t in tags_rows:
                mid = t["media_id"]
                if mid not in tags_map:
                    tags_map[mid] = []
                tags_map[mid].append(dict(t))

            for item in paged_items:
                item["faces"] = faces_map.get(item["id"], [])
                item["tags"] = tags_map.get(item["id"], [])

        return {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "results": paged_items
        }


# --- Media Metadata Lookup Functions ---

def get_all_media_metadata() -> Dict[str, Dict[str, Any]]:
    """Returns mapping of file_path and lowercase file_name to metadata with full semantic attributes."""
    with closing(get_db_connection()) as conn:
        try:
            rows = conn.execute("""
                SELECT m.file_path, m.file_name, meta.summary, meta.summary_ru, meta.description, meta.description_ru,
                       meta.environment, meta.lighting, meta.weather, meta.time_of_day, meta.ocr_text, meta.raw_gemini
                FROM media_items m
                JOIN media_metadata meta ON m.id = meta.media_id;
            """).fetchall()
            meta_map: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                data = {
                    "summary": r["summary"],
                    "summary_ru": r["summary_ru"],
                    "description": r["description"],
                    "description_ru": r["description_ru"],
                    "environment": r["environment"],
                    "lighting": r["lighting"],
                    "weather": r["weather"],
                    "time_of_day": r["time_of_day"],
                    "ocr_text": r["ocr_text"],
                }
                raw_gem = r["raw_gemini"]
                if raw_gem:
                    try:
                        g_obj = json.loads(raw_gem) if isinstance(raw_gem, str) else raw_gem
                        if isinstance(g_obj, dict):
                            data["lighting_ru"] = g_obj.get("lighting_ru")
                            data["weather_ru"] = g_obj.get("weather_ru")
                            data["time_of_day_ru"] = g_obj.get("time_of_day_ru")
                            data["exif_analysis"] = g_obj.get("exif_analysis")
                            data["exif_analysis_ru"] = g_obj.get("exif_analysis_ru")
                            data["transcription"] = g_obj.get("transcription")
                            data["transcription_ru"] = g_obj.get("transcription_ru")
                            data["timeline_events"] = g_obj.get("timeline_events")
                    except Exception:
                        pass
                if r["file_path"]:
                    meta_map[r["file_path"]] = data
                if r["file_name"]:
                    meta_map[r["file_name"].lower()] = data
            return meta_map
        except Exception:
            return {}


# --- Backward Compatibility Layer for Face Registry & Sync History ---

def get_sync_record(file_path: str) -> Optional[dict]:
    with closing(get_db_connection()) as conn:
        row = conn.execute("SELECT * FROM sync_history WHERE file_path = ?", (file_path,)).fetchone()
        return dict(row) if row else None


def update_sync_record(
    file_path: str,
    file_size: int,
    mtime: float,
    status: str,
    sidecar_path: Optional[str] = None,
    error_message: Optional[str] = None
):
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("""
                    INSERT INTO sync_history (file_path, file_size, mtime, status, sidecar_path, error_message, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
                    ON CONFLICT(file_path) DO UPDATE SET
                        file_size = excluded.file_size,
                        mtime = excluded.mtime,
                        status = excluded.status,
                        sidecar_path = excluded.sidecar_path,
                        error_message = excluded.error_message,
                        processed_at = datetime('now', 'localtime')
                """, (file_path, file_size, mtime, status, sidecar_path, error_message))


def get_all_sync_records() -> Dict[str, dict]:
    with closing(get_db_connection()) as conn:
        rows = conn.execute("SELECT * FROM sync_history").fetchall()
        return {r["file_path"]: dict(r) for r in rows}


def generate_next_face_id() -> str:
    """Generate next unique face ID identifier in format FACE_ID_XX."""
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            rows = conn.execute("SELECT face_id FROM face_registry WHERE face_id LIKE 'FACE_ID_%'").fetchall()
            max_num = 0
            for r in rows:
                match = re.search(r"FACE_ID_(\d+)", r["face_id"])
                if match:
                    try:
                        num = int(match.group(1))
                        if num > max_num:
                            max_num = num
                    except ValueError:
                        pass
            return f"FACE_ID_{max_num + 1:02d}"


def register_face(
    face_id: str,
    embedding: Optional[np.ndarray],
    name: Optional[str] = None,
    image_path: Optional[str] = None,
    confidence: Optional[float] = None,
    source_file: Optional[str] = None,
    is_reference: int = 1,
    person_id: Optional[str] = None
) -> str:
    """Register or update a face entry in face_registry."""
    with _db_write_lock:
        display_name = name or face_id
        p_id = person_id or (display_name if is_reference and not display_name.startswith("face_") else None)
        save_face_to_registry(
            face_id=face_id,
            name=display_name,
            embedding=embedding,
            image_path=image_path,
            confidence=confidence,
            source_file=source_file,
            is_reference=bool(is_reference)
        )
        if p_id and is_reference:
            with closing(get_db_connection()) as conn:
                with conn:
                    conn.execute("""
                        INSERT INTO persons (id, name)
                        VALUES (?, ?)
                        ON CONFLICT(name) DO UPDATE SET id = excluded.id;
                    """, (p_id, display_name))
                    conn.execute("UPDATE face_registry SET person_id = ? WHERE face_id = ?", (p_id, face_id))
        return display_name


def get_all_reference_faces_detailed() -> List[Dict[str, Any]]:
    """Retrieve all active reference faces with embeddings and metadata."""
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT face_id, person_id, name, embedding, image_path, confidence, source_file, created_at, is_reference
            FROM face_registry
            WHERE is_reference = 1
        """).fetchall()
        results = []
        for r in rows:
            blob = r["embedding"]
            emb = np.frombuffer(blob, dtype=np.float32) if blob else None
            results.append({
                "face_id": r["face_id"],
                "person_id": r["person_id"] or r["name"],
                "name": r["name"],
                "embedding": emb,
                "image_path": r["image_path"],
                "confidence": r["confidence"],
                "source_file": r["source_file"],
                "created_at": r["created_at"],
                "is_reference": r["is_reference"]
            })
        return results


def get_unrecognized_faces_detailed() -> List[Dict[str, Any]]:
    """Retrieve all candidate/unrecognized faces with embeddings."""
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT face_id, person_id, name, embedding, image_path, confidence, source_file, created_at, is_reference
            FROM face_registry
            WHERE is_reference = 0 OR name LIKE 'face_%'
        """).fetchall()
        results = []
        for r in rows:
            blob = r["embedding"]
            emb = np.frombuffer(blob, dtype=np.float32) if blob else None
            results.append({
                "face_id": r["face_id"],
                "person_id": r["person_id"] or r["name"],
                "name": r["name"],
                "embedding": emb,
                "image_path": r["image_path"],
                "confidence": r["confidence"],
                "source_file": r["source_file"],
                "created_at": r["created_at"],
                "is_reference": r["is_reference"]
            })
        return results


def get_all_registered_faces(only_references: bool = True) -> List[Tuple[str, str, np.ndarray]]:
    with closing(get_db_connection()) as conn:
        query = "SELECT face_id, name, embedding FROM face_registry"
        if only_references:
            query += " WHERE is_reference = 1"
        rows = conn.execute(query).fetchall()
        results = []
        for r in rows:
            blob = r["embedding"]
            emb = np.frombuffer(blob, dtype=np.float32) if blob else None
            results.append((r["face_id"], r["name"], emb))
        return results


def get_face_name_mapping() -> Dict[str, str]:
    with closing(get_db_connection()) as conn:
        rows = conn.execute("SELECT face_id, name FROM face_registry").fetchall()
        return {r["face_id"]: r["name"] for r in rows}


def save_face_to_registry(
    face_id: str,
    name: str,
    embedding: Optional[np.ndarray],
    image_path: Optional[str] = None,
    confidence: Optional[float] = None,
    source_file: Optional[str] = None,
    is_reference: bool = True
):
    emb_bytes = embedding.astype(np.float32).tobytes() if embedding is not None else None
    with _db_write_lock:
        with closing(get_db_connection()) as conn:
            with conn:
                conn.execute("""
                    INSERT INTO face_registry (face_id, person_id, name, embedding, image_path, confidence, source_file, is_reference, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
                    ON CONFLICT(face_id) DO UPDATE SET
                        name = excluded.name,
                        person_id = excluded.person_id,
                        confidence = excluded.confidence,
                        image_path = excluded.image_path,
                        is_reference = excluded.is_reference
                """, (face_id, name, name, emb_bytes, image_path, confidence, source_file, 1 if is_reference else 0))


def get_faces_count_by_source_file() -> Dict[str, int]:
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT source_file, COUNT(*) as count 
            FROM face_registry 
            WHERE source_file IS NOT NULL AND source_file != '' 
            GROUP BY source_file
        """).fetchall()
        return {r["source_file"]: r["count"] for r in rows}


def get_all_faces_by_source_file() -> Dict[str, List[Dict[str, Any]]]:
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT face_id, person_id, name, image_path, confidence, source_file, created_at, is_reference
            FROM face_registry
            WHERE source_file IS NOT NULL AND source_file != ''
            ORDER BY confidence DESC
        """).fetchall()
        result: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            sf = r["source_file"]
            if sf not in result:
                result[sf] = []
            result[sf].append({
                "face_id": r["face_id"],
                "person_id": r["person_id"] or r["name"],
                "name": r["name"],
                "image_path": r["image_path"],
                "confidence": r["confidence"],
                "source_file": r["source_file"],
                "created_at": r["created_at"],
                "is_reference": r["is_reference"]
            })
        return result


def get_faces_by_source_file(source_file: str) -> List[Dict[str, Any]]:
    with closing(get_db_connection()) as conn:
        base_name = Path(source_file).name
        # Try media_faces joined with media_items first to retrieve bounding boxes
        rows = conn.execute("""
            SELECT mf.face_id, mf.person_id, mf.name, mf.confidence, mf.bbox, mf.image_path, mf.time_start, mf.time_end,
                   m.file_path as source_file, mf.is_reference
            FROM media_faces mf
            JOIN media_items m ON mf.media_id = m.id
            WHERE m.file_path = ? OR m.file_path LIKE ?
            ORDER BY mf.confidence DESC
        """, (source_file, f"%{base_name}")).fetchall()
        if rows:
            res = []
            seen = set()
            for r in rows:
                d = dict(r)
                fid = d.get("face_id")
                if fid and fid not in seen:
                    seen.add(fid)
                    if d.get("bbox") and isinstance(d["bbox"], str):
                        try:
                            d["bbox"] = json.loads(d["bbox"])
                        except Exception:
                            pass
                    res.append(d)
            return res

        # Fallback to face_registry
        rows = conn.execute("""
            SELECT face_id, person_id, name, image_path, confidence, source_file, created_at, is_reference
            FROM face_registry
            WHERE source_file = ? OR source_file LIKE ? OR source_file LIKE ?
            ORDER BY confidence DESC
        """, (source_file, f"%{base_name}", f"%{source_file}%")).fetchall()
        return [dict(r) for r in rows]


def update_face_name(face_id: str, new_name: str):
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute("UPDATE face_registry SET name = ?, person_id = ? WHERE face_id = ?", (new_name, new_name, face_id))
            conn.execute("UPDATE media_faces SET name = ?, person_id = ? WHERE face_id = ?", (new_name, new_name, face_id))


def assign_face_to_person(face_id: str, target_name: str) -> bool:
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO persons (id, name)
                VALUES (?, ?)
                ON CONFLICT(name) DO NOTHING;
            """, (target_name, target_name))
            conn.execute("UPDATE face_registry SET name = ?, person_id = ?, is_reference = 1 WHERE face_id = ?", (target_name, target_name, face_id))
            conn.execute("UPDATE media_faces SET name = ?, person_id = ?, is_reference = 1 WHERE face_id = ?", (target_name, target_name, face_id))
            return True


def assign_group_to_person(face_ids: List[str], target_name: str) -> bool:
    if not face_ids or not target_name:
        return False
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute("""
                INSERT INTO persons (id, name)
                VALUES (?, ?)
                ON CONFLICT(name) DO NOTHING;
            """, (target_name, target_name))
            placeholders = ",".join("?" for _ in face_ids)
            conn.execute(f"UPDATE face_registry SET name = ?, person_id = ?, is_reference = 1 WHERE face_id IN ({placeholders})", [target_name, target_name] + list(face_ids))
            conn.execute(f"UPDATE media_faces SET name = ?, person_id = ?, is_reference = 1 WHERE face_id IN ({placeholders})", [target_name, target_name] + list(face_ids))
            return True


def reset_face_assignment(face_id: str) -> bool:
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute("UPDATE face_registry SET name = ?, person_id = ?, is_reference = 0 WHERE face_id = ?", (face_id, face_id, face_id))
            conn.execute("UPDATE media_faces SET name = ?, person_id = ?, is_reference = 0 WHERE face_id = ?", (face_id, face_id, face_id))
            return True


def reset_face_assignments_by_filename(filename_query: str) -> Dict[str, Any]:
    clean_query = filename_query.strip()
    if not clean_query:
        return {"reset_count": 0, "face_ids": []}
    base_name = Path(clean_query).name
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT face_id FROM face_registry
            WHERE source_file LIKE ? OR source_file LIKE ? OR source_file = ?
        """, (f"%{clean_query}%", f"%{base_name}%", clean_query)).fetchall()
        matched_face_ids = [r["face_id"] for r in rows]
        if matched_face_ids:
            with conn:
                placeholders = ",".join("?" for _ in matched_face_ids)
                conn.execute(f"UPDATE face_registry SET name = face_id, person_id = face_id, is_reference = 0 WHERE face_id IN ({placeholders})", matched_face_ids)
                conn.execute(f"UPDATE media_faces SET name = face_id, person_id = face_id, is_reference = 0 WHERE face_id IN ({placeholders})", matched_face_ids)
    return {"reset_count": len(matched_face_ids), "face_ids": matched_face_ids}


def delete_face(face_id: str) -> bool:
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute("DELETE FROM face_registry WHERE face_id = ?", (face_id,))
            conn.execute("DELETE FROM media_faces WHERE face_id = ?", (face_id,))
            return True


def add_person_to_media_file(file_path: str, person_name: str) -> Dict[str, Any]:
    """
    Manually associate a Person with a media file.
    Creates a face_registry entry and links to media_faces, as well as updating sidecar if present.
    """
    import uuid
    trimmed_name = person_name.strip()
    if not trimmed_name:
        raise ValueError("Person name cannot be empty")
    
    clean_file = file_path.strip()
    base_name = Path(clean_file).name
    face_id = f"manual_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    
    with closing(get_db_connection()) as conn:
        with conn:
            # 1. Ensure person is registered in persons table
            conn.execute("""
                INSERT INTO persons (id, name)
                VALUES (?, ?)
                ON CONFLICT(name) DO NOTHING;
            """, (trimmed_name, trimmed_name))
            
            # 2. Check if this person already has a reference image in face_registry
            sample_img_row = conn.execute("""
                SELECT image_path, embedding FROM face_registry
                WHERE name = ? AND image_path IS NOT NULL AND image_path != ''
                LIMIT 1
            """, (trimmed_name,)).fetchone()
            
            ref_image = sample_img_row["image_path"] if sample_img_row else None
            ref_emb = sample_img_row["embedding"] if sample_img_row else None
            
            # 3. Insert into face_registry
            conn.execute("""
                INSERT INTO face_registry (
                    face_id, person_id, name, embedding, image_path, confidence, source_file, is_reference, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, datetime('now', 'localtime'))
            """, (face_id, trimmed_name, trimmed_name, ref_emb, ref_image, 1.0, clean_file))
            
            # 4. Link into media_faces if media_items has record for this file
            m_row = conn.execute("""
                SELECT id FROM media_items 
                WHERE file_path = ? OR file_path LIKE ? OR file_name = ?
                LIMIT 1
            """, (clean_file, f"%{base_name}", base_name)).fetchone()
            
            media_id = m_row["id"] if m_row else None
            if media_id:
                conn.execute("""
                    INSERT INTO media_faces (
                        media_id, face_id, person_id, name, confidence, image_path, is_reference
                    )
                    VALUES (?, ?, ?, ?, 1.0, ?, 1)
                """, (media_id, face_id, trimmed_name, trimmed_name, ref_image))
                _refresh_fts_entry(conn, media_id)

    # 5. Also update sidecar JSON on disk if it exists
    try:
        sidecar_candidates = [
            config.OUTPUT_FOLDER / f"{base_name}.json",
            Path(clean_file).parent / f"{base_name}.json"
        ]
        for sc in sidecar_candidates:
            if sc.is_file():
                with open(sc, "r", encoding="utf-8", errors="ignore") as f:
                    sdata = json.load(f)
                
                faces = sdata.get("faces", [])
                if not isinstance(faces, list):
                    faces = []
                faces.append({
                    "face_id": face_id,
                    "name": trimmed_name,
                    "confidence": 1.0,
                    "is_reference": True,
                    "image_path": ref_image
                })
                sdata["faces"] = faces
                
                face_names = sdata.get("face_names", [])
                if isinstance(face_names, list) and trimmed_name not in face_names:
                    face_names.append(trimmed_name)
                    sdata["face_names"] = face_names
                
                with open(sc, "w", encoding="utf-8") as f:
                    json.dump(sdata, f, indent=2, ensure_ascii=False)
                break
    except Exception as e:
        logger.warning(f"Failed to update sidecar for manual person tag: {e}")

    return {
        "face_id": face_id,
        "person_id": trimmed_name,
        "name": trimmed_name,
        "image_path": ref_image,
        "confidence": 1.0,
        "source_file": clean_file,
        "is_reference": 1
    }


def remove_face_from_media_file(file_path: str, face_id: str) -> bool:
    """
    Remove or unlink a face / person association from a media file.
    """
    clean_file = file_path.strip()
    base_name = Path(clean_file).name
    
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute("DELETE FROM face_registry WHERE face_id = ?", (face_id,))
            conn.execute("DELETE FROM media_faces WHERE face_id = ?", (face_id,))
            
            m_row = conn.execute("""
                SELECT id FROM media_items 
                WHERE file_path = ? OR file_path LIKE ? OR file_name = ?
                LIMIT 1
            """, (clean_file, f"%{base_name}", base_name)).fetchone()
            if m_row:
                _refresh_fts_entry(conn, m_row["id"])

    try:
        sidecar_candidates = [
            config.OUTPUT_FOLDER / f"{base_name}.json",
            Path(clean_file).parent / f"{base_name}.json"
        ]
        for sc in sidecar_candidates:
            if sc.is_file():
                with open(sc, "r", encoding="utf-8", errors="ignore") as f:
                    sdata = json.load(f)
                
                if "faces" in sdata and isinstance(sdata["faces"], list):
                    sdata["faces"] = [f for f in sdata["faces"] if f.get("face_id") != face_id]
                
                with open(sc, "w", encoding="utf-8") as f:
                    json.dump(sdata, f, indent=2, ensure_ascii=False)
                break
    except Exception as e:
        logger.warning(f"Failed to update sidecar when removing face: {e}")

    return True


def get_unrecognized_faces() -> List[Dict[str, Any]]:
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT face_id, name, image_path, confidence, source_file, created_at, is_reference
            FROM face_registry
            WHERE (is_reference = 0 OR name LIKE 'face_%')
            ORDER BY created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_unrecognized_face_groups(similarity_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    threshold = similarity_threshold if similarity_threshold is not None else config.FACE_SIMILARITY_THRESHOLD
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT face_id, name, embedding, image_path, confidence, source_file, created_at, is_reference
            FROM face_registry
            WHERE (is_reference = 0 OR name LIKE 'face_%') AND embedding IS NOT NULL
            ORDER BY created_at DESC
        """).fetchall()

    if not rows:
        return []

    items = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
        items.append({
            "face_id": r["face_id"],
            "name": r["name"],
            "image_path": r["image_path"],
            "confidence": r["confidence"],
            "source_file": r["source_file"],
            "created_at": r["created_at"],
            "is_reference": r["is_reference"],
            "embedding": emb
        })

    clusters = []
    used = set()
    for i, it in enumerate(items):
        if it["face_id"] in used or it["embedding"] is None:
            continue
        group_members = [it]
        used.add(it["face_id"])
        for j in range(i + 1, len(items)):
            other = items[j]
            if other["face_id"] in used or other["embedding"] is None:
                continue
            norm_a = np.linalg.norm(it["embedding"])
            norm_b = np.linalg.norm(other["embedding"])
            if norm_a > 0 and norm_b > 0:
                cos_sim = float(np.dot(it["embedding"], other["embedding"]) / (norm_a * norm_b))
                if cos_sim >= threshold:
                    group_members.append(other)
                    used.add(other["face_id"])

        clean_members = []
        for m in group_members:
            m_copy = dict(m)
            m_copy.pop("embedding", None)
            clean_members.append(m_copy)

        clusters.append({
            "group_id": f"group_{it['face_id']}",
            "representative": clean_members[0],
            "count": len(clean_members),
            "members": clean_members
        })
    return clusters


def get_known_persons() -> List[Dict[str, Any]]:
    with closing(get_db_connection()) as conn:
        rows = conn.execute("""
            SELECT 
                face_id,
                COALESCE(person_id, name) as person_id,
                name,
                image_path,
                confidence,
                source_file,
                created_at
            FROM face_registry
            WHERE is_reference = 1 AND name NOT LIKE 'face_%' AND name != ''
            ORDER BY name ASC, created_at ASC
        """).fetchall()
        
        persons_map: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            p_name = r["name"]
            if p_name not in persons_map:
                persons_map[p_name] = {
                    "person_id": r["person_id"],
                    "name": p_name,
                    "reference_count": 0,
                    "reference_face_count": 0,
                    "reference_faces": [],
                    "sample_images": []
                }
            
            face_item = {
                "face_id": r["face_id"],
                "image_path": r["image_path"],
                "confidence": r["confidence"],
                "source_file": r["source_file"],
                "created_at": r["created_at"]
            }
            persons_map[p_name]["reference_faces"].append(face_item)
            persons_map[p_name]["reference_count"] += 1
            persons_map[p_name]["reference_face_count"] += 1
            if r["image_path"]:
                persons_map[p_name]["sample_images"].append(r["image_path"])

        return list(persons_map.values())
