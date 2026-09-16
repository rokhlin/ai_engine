"""
Automated Database Migration & Backup Engine for Media Cataloger.
Provides safe, zero-downtime schema evolution, pre-migration hot backups,
version tracking via 'schema_migrations' table, and helper utilities.
"""

import sqlite3
import time
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from contextlib import closing
from datetime import datetime

logger = logging.getLogger(__name__)

# Registry of versioned migrations: (version: int, name: str, up_func: Callable[[sqlite3.Connection], None])
MIGRATIONS: List[Dict[str, Any]] = []


def register_migration(version: int, name: str):
    """Decorator to register a schema migration step."""
    def decorator(func: Callable[[sqlite3.Connection], None]):
        MIGRATIONS.append({
            "version": version,
            "name": name,
            "func": func
        })
        # Keep migrations sorted by version
        MIGRATIONS.sort(key=lambda m: m["version"])
        return func
    return decorator


def backup_database(db_path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Perform a safe, atomic hot backup of the SQLite database before applying migrations.
    Uses SQLite's online backup API to ensure integrity even if active readers are connected.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        return None

    if backup_dir is None:
        backup_dir = db_path.parent / "backups"
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"{db_path.stem}_{timestamp}.bak{db_path.suffix}"

    try:
        # Use sqlite3 online backup API for safe hot copy
        src_conn = sqlite3.connect(str(db_path), timeout=30.0)
        dst_conn = sqlite3.connect(str(backup_file), timeout=30.0)
        with src_conn, dst_conn:
            src_conn.backup(dst_conn, pages=100)
        src_conn.close()
        dst_conn.close()
        logger.info(f"Database safety backup created at: {backup_file}")
        return backup_file
    except Exception as e:
        logger.warning(f"Online SQLite backup failed ({e}), falling back to file copy.")
        try:
            shutil.copy2(db_path, backup_file)
            logger.info(f"Database safety backup created (copy) at: {backup_file}")
            return backup_file
        except Exception as copy_err:
            logger.error(f"Failed to create database backup: {copy_err}")
            return None


def get_applied_migrations(conn: sqlite3.Connection) -> List[int]:
    """Retrieve list of already applied migration version numbers."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            duration_ms REAL
        );
    """)
    cur = conn.cursor()
    cur.execute("SELECT version FROM schema_migrations ORDER BY version ASC;")
    rows = cur.fetchall()
    return [row[0] for row in rows]


def add_column_if_not_exists(conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str):
    """
    Safely add a column to an existing table if it does not already exist.
    Prevents SQLite 'duplicate column name' errors.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1].lower() for col in cur.fetchall()]
    if column_name.lower() not in columns:
        logger.info(f"Adding column '{column_name}' ({column_type}) to table '{table_name}'...")
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};")
        return True
    return False


def run_migrations(db_path: Path, auto_backup: bool = True) -> Dict[str, Any]:
    """
    Execute all registered pending migrations in chronological order inside transactions.
    Creates a pre-migration backup before applying any new changes.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    applied_count = 0
    applied_list = []
    backup_path = None

    # Connect to database
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        with conn:
            # Enable WAL mode & foreign keys
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")

            already_applied = set(get_applied_migrations(conn))
            pending = [m for m in MIGRATIONS if m["version"] not in already_applied]

            if pending and auto_backup and db_path.is_file() and db_path.stat().st_size > 0:
                backup_path = backup_database(db_path)

            for migration in pending:
                v = migration["version"]
                name = migration["name"]
                logger.info(f"Applying migration {v:03d}_{name}...")
                start_t = time.perf_counter()

                # Execute migration step
                migration["func"](conn)

                dur_ms = round((time.perf_counter() - start_t) * 1000, 2)
                conn.execute(
                    "INSERT INTO schema_migrations (version, name, applied_at, duration_ms) VALUES (?, ?, datetime('now', 'localtime'), ?);",
                    (v, name, dur_ms)
                )
                applied_count += 1
                applied_list.append({"version": v, "name": name, "duration_ms": dur_ms})
                logger.info(f"Migration {v:03d}_{name} applied successfully in {dur_ms}ms.")

    finally:
        conn.close()

    return {
        "status": "success",
        "applied_count": applied_count,
        "applied_migrations": applied_list,
        "backup_path": str(backup_path) if backup_path else None
    }


def get_migration_status(db_path: Path) -> Dict[str, Any]:
    """Inspect current migration status, applied versions, and pending migrations."""
    db_path = Path(db_path)
    if not db_path.is_file():
        return {
            "status": "not_found",
            "db_path": str(db_path),
            "applied_count": 0,
            "pending_count": len(MIGRATIONS),
            "applied": [],
            "pending": [{"version": m["version"], "name": m["name"]} for m in MIGRATIONS]
        }

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        applied_versions = get_applied_migrations(conn)
        applied_map = {row[0]: (row[1], row[2], row[3]) for row in conn.execute("SELECT version, name, applied_at, duration_ms FROM schema_migrations;").fetchall()}
        
        applied_details = []
        for v in applied_versions:
            info = applied_map.get(v, (f"migration_{v}", "unknown", 0.0))
            applied_details.append({
                "version": v,
                "name": info[0],
                "applied_at": info[1],
                "duration_ms": info[2]
            })

        applied_set = set(applied_versions)
        pending = [{"version": m["version"], "name": m["name"]} for m in MIGRATIONS if m["version"] not in applied_set]

        return {
            "status": "ready",
            "db_path": str(db_path),
            "applied_count": len(applied_details),
            "pending_count": len(pending),
            "latest_version": max(applied_versions) if applied_versions else 0,
            "applied": applied_details,
            "pending": pending
        }
    finally:
        conn.close()


# =====================================================================
# Registered Migration Steps (Chronological Version History)
# =====================================================================

@register_migration(1, "initial_core_schema")
def migration_001_initial_core_schema(conn: sqlite3.Connection):
    """
    Migration 001: Baseline schema for Media Cataloger.
    Creates sync_history, persons, face_registry, media_items,
    media_metadata, media_faces, media_tags, video_timeline_events,
    media_embeddings, and media_fts full-text virtual table.
    """
    # 1. Sync History
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_history (
            file_path TEXT PRIMARY KEY,
            file_size INTEGER NOT NULL,
            mtime REAL NOT NULL,
            status TEXT NOT NULL,
            processed_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            sidecar_path TEXT,
            error_message TEXT
        );
    """)
    add_column_if_not_exists(conn, "sync_history", "processed_at", "TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))")
    add_column_if_not_exists(conn, "sync_history", "sidecar_path", "TEXT")
    add_column_if_not_exists(conn, "sync_history", "error_message", "TEXT")

    # 2. Persons Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
    """)

    # 3. Face Registry
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT UNIQUE NOT NULL,
            person_id TEXT,
            name TEXT NOT NULL,
            embedding BLOB,
            image_path TEXT,
            confidence REAL,
            source_file TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            is_reference INTEGER NOT NULL DEFAULT 1
        );
    """)
    add_column_if_not_exists(conn, "face_registry", "person_id", "TEXT")
    add_column_if_not_exists(conn, "face_registry", "image_path", "TEXT")
    add_column_if_not_exists(conn, "face_registry", "confidence", "REAL")
    add_column_if_not_exists(conn, "face_registry", "source_file", "TEXT")
    add_column_if_not_exists(conn, "face_registry", "created_at", "TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))")
    add_column_if_not_exists(conn, "face_registry", "is_reference", "INTEGER NOT NULL DEFAULT 1")

    # 4. Core Media Items
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media_items (
            id TEXT PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            file_name TEXT NOT NULL,
            media_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mtime REAL,
            duration REAL,
            media_date TEXT,
            phash TEXT,
            status TEXT NOT NULL DEFAULT 'PROCESSED',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON media_items(media_type);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_date ON media_items(media_date DESC);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_filename ON media_items(file_name);")

    # 5. Media Metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media_metadata (
            media_id TEXT PRIMARY KEY,
            summary TEXT,
            summary_ru TEXT,
            description TEXT,
            description_ru TEXT,
            environment TEXT,
            lighting TEXT,
            weather TEXT,
            time_of_day TEXT,
            ocr_text TEXT,
            camera_make TEXT,
            camera_model TEXT,
            latitude REAL,
            longitude REAL,
            location_name TEXT,
            raw_exif TEXT,
            raw_gemini TEXT,
            raw_defects TEXT,
            FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_camera ON media_metadata(camera_make, camera_model);")

    # 6. Media Faces
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT,
            face_id TEXT NOT NULL,
            person_id TEXT,
            name TEXT NOT NULL,
            confidence REAL,
            bbox TEXT,
            image_path TEXT,
            time_start REAL,
            time_end REAL,
            time_intervals TEXT,
            is_reference INTEGER DEFAULT 1,
            FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
        );
    """)
    add_column_if_not_exists(conn, "media_faces", "person_id", "TEXT")
    add_column_if_not_exists(conn, "media_faces", "bbox", "TEXT")
    add_column_if_not_exists(conn, "media_faces", "image_path", "TEXT")
    add_column_if_not_exists(conn, "media_faces", "time_start", "REAL")
    add_column_if_not_exists(conn, "media_faces", "time_end", "REAL")
    add_column_if_not_exists(conn, "media_faces", "time_intervals", "TEXT")
    add_column_if_not_exists(conn, "media_faces", "is_reference", "INTEGER DEFAULT 1")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_faces_person ON media_faces(person_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_faces_media ON media_faces(media_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_faces_name ON media_faces(name);")

    # 7. Media Tags
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT,
            tag TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            confidence REAL DEFAULT 1.0,
            UNIQUE(media_id, tag, category),
            FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_tags_tag ON media_tags(tag);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_tags_category ON media_tags(category);")

    # 8. Video Timeline Events
    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_timeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT,
            timestamp_start TEXT,
            timestamp_end TEXT,
            start_sec REAL,
            end_sec REAL,
            activity TEXT,
            activity_ru TEXT,
            FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_video_events_media ON video_timeline_events(media_id);")

    # 9. Vector Embeddings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media_embeddings (
            media_id TEXT PRIMARY KEY,
            text_embedding BLOB,
            indexed_text TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
        );
    """)

    # 10. FTS5 Virtual Table
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
                media_id UNINDEXED,
                file_name,
                summary,
                description,
                ocr_text,
                tags,
                faces,
                location,
                tokenize = 'porter unicode61'
            );
        """)
    except Exception as e:
        logger.warning(f"FTS5 table initialization notice: {e}")
