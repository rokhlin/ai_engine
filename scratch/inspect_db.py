import sqlite3
import os
import sys

paths = [
    r"\\ZIMABOARD\sda1\media_cataloger\config\catalog_history.db",
    r"Y:\catalog_history.db",
    r"C:\Users\rokhl\.gemini\antigravity\scratch\media_cataloger\catalog_history.db",
    r"C:\Users\rokhl\.gemini\antigravity\scratch\media_cataloger\data\config\catalog_history.db"
]

for p in paths:
    print(f"\n==================== Path: {p} (Exists: {os.path.exists(p)}) ====================")
    if os.path.exists(p):
        try:
            conn = sqlite3.connect(p)
            conn.row_factory = sqlite3.Row
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            print("Tables:", tables)
            for t in tables:
                if not t.startswith("sqlite") and "fts" not in t:
                    cnt = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                    print(f"  - {t}: {cnt} rows")
            if "sync_history" in tables:
                print("Recent sync_history rows:")
                for r in conn.execute("SELECT file_path, status, error_message, processed_at FROM sync_history ORDER BY processed_at DESC LIMIT 10").fetchall():
                    print("   ", dict(r))
        except Exception as e:
            print("Error reading db:", e)
