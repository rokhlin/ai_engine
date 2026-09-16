import sqlite3
import os
import sys
from src.utils.gpu_duplicates import run_gpu_duplicate_clustering

db_path = "Y:/catalog_history.db"
con = sqlite3.connect(db_path)
cur = con.cursor()
rows = cur.execute("SELECT file_path FROM media_items WHERE is_vault = 0 OR is_vault IS NULL").fetchall()
files = [{"file_path": r[0]} for r in rows]

print(f"Total files in DB: {len(files)}")
print(f"Checking if files exist on disk:")
exist_count = sum(1 for f in files if os.path.exists(f["file_path"]))
print(f"Existing files on disk: {exist_count} / {len(files)}")

for f in files[:10]:
    print("  ", f["file_path"], "-> exists:", os.path.exists(f["file_path"]))

res = run_gpu_duplicate_clustering(files, mode="all", similarity_threshold=0.9, burst_window_seconds=3.0)
print("Result:")
print("  Engine:", res.get("engine"))
print("  Elapsed:", res.get("elapsed_seconds"))
print("  Scanned:", res.get("scanned_files_count"))
print("  Total groups:", res.get("total_groups"))
print("  Groups:", len(res.get("groups", [])))
