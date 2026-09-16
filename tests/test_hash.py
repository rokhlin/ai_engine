import pytest
from src.utils import hash

def test_get_hamming_distance():
    # Identical hashes
    h1 = "0000000000000000"
    assert hash.get_hamming_distance(h1, h1) == 0
    
    # Differ by 1 nibble in hex (last digit f is 1111, differs from 0 by 4 bits)
    h2 = "000000000000000f"
    assert hash.get_hamming_distance(h1, h2) == 4
    
    # Completely different hashes
    h3 = "ffffffffffffffff"
    assert hash.get_hamming_distance(h1, h3) == 64

def test_cluster_duplicates():
    # Mock files metadata
    files_meta = [
        # Burst series 1 (duplicates)
        {
            "file_path": "photoA.jpg",
            "datetime": "2023:08:20 14:00:00", # t = 0
            "phash": "0000000000000000"
        },
        {
            "file_path": "photoB.jpg",
            "datetime": "2023:08:20 14:00:05", # t = 5 (diff 5s)
            "phash": "0000000000000000"
        },
        {
            "file_path": "photoC.jpg",
            "datetime": "2023:08:20 14:00:12", # t = 12 (diff 7s from B, 12s from A)
            "phash": "000000000000000f" # dist = 4
        },
        # Single file (different hash)
        {
            "file_path": "photoD.jpg",
            "datetime": "2023:08:20 14:00:15", # t = 15
            "phash": "ffffffffffffffff"
        },
        # Single file (same hash, but taken much later)
        {
            "file_path": "photoE.jpg",
            "datetime": "2023:08:20 14:05:00", # t = 300
            "phash": "0000000000000000"
        }
    ]
    
    mapping = hash.cluster_duplicates(files_meta, max_time_diff=15.0, max_hash_dist=12)
    
    # Exactly one duplicate group should be created
    assert "photoA.jpg" in mapping
    assert "photoB.jpg" in mapping
    assert "photoC.jpg" in mapping
    
    # All three files must share the same group_id
    group_id = mapping["photoA.jpg"]
    assert mapping["photoB.jpg"] == group_id
    assert mapping["photoC.jpg"] == group_id
    
    # photoD.jpg and photoE.jpg should not be in duplicate group
    assert "photoD.jpg" not in mapping
    assert "photoE.jpg" not in mapping

