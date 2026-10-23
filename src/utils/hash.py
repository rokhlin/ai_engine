import hashlib
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Set, Any, Union

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


def calculate_dhash(image_path: Union[Path, str]) -> Optional[str]:
    """Calculate 64-bit difference hash (dHash) with auto EXIF orientation.
    Returns 16-character lowercase hex string.
    """
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.int16)
            diff = (arr[:, :-1] > arr[:, 1:]).astype(np.uint8).flatten()
            bits = diff.tolist()
            return "".join(
                format(int("".join(map(str, bits[i:i+4])), 2), "x")
                for i in range(0, 64, 4)
            )
    except Exception as e:
        print(f"Error calculating dHash for {image_path}: {e}")
        return None


def calculate_phash(image_path: Union[Path, str]) -> Optional[str]:
    """Canonical perceptual hash. Standardized to 64-bit dHash for full parity with web engine."""
    return calculate_dhash(image_path)


def parse_datetime(dt_str: Optional[str]) -> Optional[float]:
    """Safely parse EXIF date-time string to timestamp."""
    if not dt_str:
        return None
    dt_str = str(dt_str).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            clean_str = dt_str[:19].replace("-", ":").replace("T", " ")
            dt = datetime.strptime(clean_str, "%Y:%m:%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            continue
    return None


def get_hamming_distance(hash1_str: str, hash2_str: str) -> int:
    """Calculate Hamming distance between two 16-character hex hashes using fast bitwise XOR popcount."""
    if not hash1_str or not hash2_str or len(hash1_str) != 16 or len(hash2_str) != 16:
        return 64
    try:
        val = int(hash1_str, 16) ^ int(hash2_str, 16)
        return bin(val).count("1")
    except Exception:
        return 64


def cluster_duplicates(
    files_meta: List[Dict[str, Any]], 
    max_time_diff: float = 15.0, 
    max_hash_dist: int = 12
) -> Dict[str, str]:
    """Group similar files based on capture time and perceptual hash.
    Per SDM BUG-020 rules: only files with valid EXIF datetime are candidates for series.
    """
    valid_files = []
    for f in files_meta:
        ts = parse_datetime(f.get("datetime"))
        if ts is not None and f.get("phash"):
            valid_files.append({
                "file_path": f["file_path"],
                "timestamp": ts,
                "phash": f["phash"]
            })
            
    valid_files.sort(key=lambda x: x["timestamp"])
    
    n = len(valid_files)
    adj: Dict[str, Set[str]] = {f["file_path"]: set() for f in valid_files}
    
    for i in range(n):
        f1 = valid_files[i]
        path1 = f1["file_path"]
        t1 = f1["timestamp"]
        h1 = f1["phash"]
        
        for j in range(i + 1, n):
            f2 = valid_files[j]
            t2 = f2["timestamp"]
            
            if t2 - t1 > max_time_diff:
                break
                
            path2 = f2["file_path"]
            h2 = f2["phash"]
            
            dist = get_hamming_distance(h1, h2)
            if dist <= max_hash_dist:
                adj[path1].add(path2)
                adj[path2].add(path1)
                
    visited: Set[str] = set()
    groups: List[List[str]] = []
    
    for f in valid_files:
        path = f["file_path"]
        if path not in visited:
            component = []
            queue = [path]
            visited.add(path)
            
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(component) >= 2:
                groups.append(component)
                
    file_to_group = {}
    for idx, group in enumerate(groups):
        group_id = f"dup_group_{idx + 1:03d}"
        for path in group:
            file_to_group[path] = group_id
            
    return file_to_group
