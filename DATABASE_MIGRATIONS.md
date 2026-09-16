# Database Migrations & Production Persistence Guide

This document outlines the architecture, standard operating procedures, and safety mechanisms for evolving the database schema in **Media Cataloger** while keeping production data 100% safe, persistent, and backward-compatible.

---

## 📌 1. Production Persistence Architecture

In production environments (ZimaOS, CasaOS, Unraid, TrueNAS, or generic Docker hosts), the SQLite database (`catalog_history.db`) stores:
- **Media Ingestion & Sync History** (`sync_history`, `media_items`)
- **AI Semantics & EXIF Metadata** (`media_metadata`)
- **Face Recognition Embeddings & Crop References** (`face_registry`, `media_faces`)
- **Person Profiles & Family Tree Links** (`persons`)
- **Media Tags & Video Timelines** (`media_tags`, `video_timeline_events`)
- **Full-Text & Vector Search Embeddings** (`media_fts`, `media_embeddings`)
- **Migration Execution History** (`schema_migrations`)

### Persistent Volume Mapping

The database resides in the directory specified by `OUTPUT_FOLDER` (default `./media_output/catalog_history.db` or `DB_PATH`).
In Docker compose setups (`docker-compose.all.yml`, `docker-compose.cataloger.yml`, `docker-compose.frontend.yml`), this folder is bound to a host directory:

```yaml
volumes:
  - ${MEDIA_OUTPUT:-/DATA/AppData/media-cataloger/output}:/app/media_output
```

> [!IMPORTANT]
> **Never store production databases inside container ephemeral layers.** Always verify that `MEDIA_OUTPUT` maps to a persistent host storage volume.

---

## 🛡️ 2. The Golden Rules of Schema Evolution

When modifying the database schema for future features:

1. **NEVER use destructive operations (`DROP TABLE`, `DROP COLUMN`) directly on production.**
2. **Always write non-breaking changes first (Expand and Contract Pattern):**
   - **Phase 1 (Expand)**: Add new nullable columns or tables with default values. Both old and new application versions continue to work without errors.
   - **Phase 2 (Migrate/Backfill)**: Backfill data in the new columns via background tasks or migration scripts.
   - **Phase 3 (Contract)**: Only after all services have updated, stop querying deprecated columns.
3. **Register all changes in `src/migrations.py`**:
   - Every schema change must be assigned an incremental integer version (`2`, `3`, `4`, ...).
   - Migrations run automatically on container startup or manually via `python manage.py db:migrate`.
4. **Pre-Migration Safety Backups are Automatic**:
   - The migration runner automatically creates an atomic hot-backup (`catalog_history.db.bak_<timestamp>`) in `media_output/backups/` before any new migration executes.

---

## 🚀 3. How to Add a New Database Migration

To introduce a new table, column, index, or data transformation:

### Step 1: Define Migration in `src/migrations.py`

Open [`src/migrations.py`](file:///c:/Users/rokhl/.gemini/antigravity/scratch/media_cataloger/src/migrations.py) and register the next sequential version number using the `@register_migration` decorator:

```python
from src.migrations import register_migration, add_column_if_not_exists

@register_migration(2, "add_camera_lens_and_color_palette")
def migration_002_add_camera_lens_and_palette(conn: sqlite3.Connection):
    """
    Migration 002: Adds lens info and dominant color palette to media_metadata.
    """
    # 1. Safely add new columns without failing if they already exist
    add_column_if_not_exists(conn, "media_metadata", "lens_model", "TEXT")
    add_column_if_not_exists(conn, "media_metadata", "color_palette", "TEXT")

    # 2. Create new indexes if needed
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_lens ON media_metadata(lens_model);")

    # 3. Create any new tables if applicable
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT NOT NULL,
            user_id TEXT DEFAULT 'default',
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (media_id) REFERENCES media_items(id) ON DELETE CASCADE
        );
    """)
```

### Step 2: Keep Frontend and Backend in Sync

If the change affects tables shared between Machine A (NestJS Frontend) and Machine B (Python Cataloger):
1. Update `createTables()` in [`frontend/server/database/database.service.ts`](file:///c:/Users/rokhl/.gemini/antigravity/scratch/media_cataloger/frontend/server/database/database.service.ts) to include the new columns/tables for fresh installs.
2. Update TypeScript interfaces and DTOs to support the new fields.

### Step 3: Write an Automated Test

Add a test case in [`tests/test_migrations.py`](file:///c:/Users/rokhl/.gemini/antigravity/scratch/media_cataloger/tests/test_migrations.py) to ensure:
- The migration runs cleanly on a database created with previous versions.
- Existing data in rows remains intact.
- Re-running the migration is idempotent (causes no errors).

---

## 🛠️ 4. Database Management CLI Commands

Use `manage.py` to inspect, back up, and migrate databases locally or inside containers:

### 1. Check Database & Migration Status
```bash
python manage.py db:status
```
Outputs:
- Active database path and file size
- SQLite journal mode and WAL settings
- List of applied migrations with execution durations
- List of pending migrations
- Row counts for all major tables

### 2. Create a Hot Database Backup
```bash
python manage.py db:backup
```
- Performs a zero-downtime atomic backup using SQLite's online backup API.
- Stores the snapshot in `media_output/backups/catalog_history_<timestamp>.bak.db`.

### 3. Run Pending Migrations Manually
```bash
python manage.py db:migrate
```
- Creates a safety pre-migration backup.
- Executes all pending migrations in order within transactions.
- Records entries in `schema_migrations`.

---

## 🔄 5. Disaster Recovery & Rollback Runbook

If an application update or faulty migration fails in production:

### Step 1: Identify the Latest Backup
Backups are located in the `backups/` folder inside your persistent media output directory:
```bash
ls -lh /DATA/AppData/media-cataloger/output/backups/
# Example output:
# catalog_history_20260829_140000.bak.db
```

### Step 2: Stop Running Containers
```bash
docker compose -f docker-compose.all.yml down
# or on Machine A / Machine B:
python manage.py down
```

### Step 3: Restore the Database Snapshot
```bash
# On the host machine:
cd /DATA/AppData/media-cataloger/output
cp backups/catalog_history_20260829_140000.bak.db catalog_history.db
```

### Step 4: Restart Containers
```bash
docker compose -f docker-compose.all.yml up -d
```

### Step 5: Verify Integrity
```bash
python manage.py db:status
```

---

## 📋 6. Checklist for Production Upgrades

Before pushing a database schema change to Git:
- [ ] Registered next migration number in `src/migrations.py` with `@register_migration`.
- [ ] Used `add_column_if_not_exists` or `CREATE TABLE IF NOT EXISTS` for all DDL statements.
- [ ] Ensured all new columns are nullable or have sensible `DEFAULT` values.
- [ ] Tested migration on a sample database with existing records (`pytest tests/test_migrations.py`).
- [ ] Updated TypeScript definitions in `frontend/server/` if API responses are altered.
- [ ] Verified `python manage.py db:status` reflects clean schema state.
