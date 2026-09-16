import sqlite3
import pytest
from pathlib import Path
from src import migrations
import manage

def test_migrations_fresh_db(tmp_path):
    db_file = tmp_path / "test_fresh.db"
    res = migrations.run_migrations(db_file, auto_backup=False)
    
    assert res["status"] == "success"
    assert res["applied_count"] >= 1
    
    # Check tables exist
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    
    assert "schema_migrations" in tables
    assert "media_items" in tables
    assert "media_metadata" in tables
    assert "persons" in tables
    assert "face_registry" in tables
    assert "sync_history" in tables

def test_add_column_if_not_exists(tmp_path):
    db_file = tmp_path / "test_columns.db"
    migrations.run_migrations(db_file, auto_backup=False)
    
    conn = sqlite3.connect(str(db_file))
    # Add new column
    added = migrations.add_column_if_not_exists(conn, "media_metadata", "lens_profile", "TEXT")
    assert added is True
    
    # Second time should be False (no error)
    added_again = migrations.add_column_if_not_exists(conn, "media_metadata", "lens_profile", "TEXT")
    assert added_again is False
    
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(media_metadata);")
    cols = [r[1] for r in cur.fetchall()]
    assert "lens_profile" in cols
    conn.close()

def test_backup_database(tmp_path):
    db_file = tmp_path / "test_backup.db"
    migrations.run_migrations(db_file, auto_backup=False)
    
    # Insert test record
    conn = sqlite3.connect(str(db_file))
    conn.execute("INSERT INTO persons (id, name) VALUES ('p1', 'Alice');")
    conn.commit()
    conn.close()
    
    # Perform backup
    backup_file = migrations.backup_database(db_file, backup_dir=tmp_path / "backups")
    assert backup_file is not None
    assert backup_file.is_file()
    assert backup_file.stat().st_size > 0
    
    # Verify backup contains the record
    b_conn = sqlite3.connect(str(backup_file))
    cur = b_conn.cursor()
    cur.execute("SELECT name FROM persons WHERE id='p1';")
    row = cur.fetchone()
    b_conn.close()
    assert row is not None
    assert row[0] == "Alice"

def test_migration_status(tmp_path):
    db_file = tmp_path / "test_status.db"
    
    status_before = migrations.get_migration_status(db_file)
    assert status_before["status"] == "not_found"
    assert status_before["pending_count"] >= 1
    
    migrations.run_migrations(db_file, auto_backup=False)
    
    status_after = migrations.get_migration_status(db_file)
    assert status_after["status"] == "ready"
    assert status_after["applied_count"] >= 1
    assert status_after["pending_count"] == 0

def test_manage_db_commands(tmp_path, monkeypatch, capsys):
    test_db = tmp_path / "catalog_history.db"
    
    # Monkeypatch get_db_path in manage
    monkeypatch.setattr(manage, "get_db_path", lambda env: test_db)
    
    env_file = tmp_path / ".env"
    env_file.write_text(f"DB_PATH={test_db}\n", encoding="utf-8")
    
    # 1. Run db:migrate
    ret_migrate = manage.cmd_db_migrate(env_file, [])
    assert ret_migrate == 0
    assert test_db.is_file()
    
    # 2. Run db:status
    ret_status = manage.cmd_db_status(env_file, [])
    assert ret_status == 0
    captured = capsys.readouterr()
    assert "Media Cataloger Database Status" in captured.out
    assert "Migration Status:" in captured.out
    
    # 3. Run db:backup
    ret_backup = manage.cmd_db_backup(env_file, [])
    assert ret_backup == 0
