import sqlite3
import json

db_path = "Y:/catalog_history.db"
con = sqlite3.connect(db_path)
cur = con.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

if "media_items" in tables:
    c = cur.execute("SELECT count(*) FROM media_items").fetchone()[0]
    print("media_items count:", c)
    samples = cur.execute("SELECT file_path, folder, phash, is_vault FROM media_items LIMIT 5").fetchall()
    print("samples:")
    for s in samples:
        print(" ", s)

if "media_hashes" in tables:
    c = cur.execute("SELECT count(*) FROM media_hashes").fetchone()[0]
    print("media_hashes count:", c)
