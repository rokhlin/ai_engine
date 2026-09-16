import imagehash
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Set, Any

def calculate_phash(image_path: Path) -> Optional[str]:
    """Calculate perceptual hash (pHash) of an image."""
    try:
        with Image.open(image_path) as img:
            h = imagehash.phash(img)
            return str(h)
    except Exception as e:
        print(f"Error calculating pHash for {image_path.name}: {e}")
        return None

def parse_datetime(dt_str: Optional[str]) -> Optional[float]:
    """Safely parse date-time string to timestamp."""
    if not dt_str:
        return None
    dt_str = dt_str.strip()
    # Standard EXIF formats: "2023:08:20 14:30:15"
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            # Strip milliseconds/timezones for basic parsing
            clean_str = dt_str[:19].replace("-", ":").replace("T", " ")
            dt = datetime.strptime(clean_str, "%Y:%m:%d %H:%M:%S")
            return dt.timestamp()
        except ValueError:
            continue
    return None

def get_hamming_distance(hash1_str: str, hash2_str: str) -> int:
    """Calculate Hamming distance between two pHash strings."""
    try:
        h1 = imagehash.hex_to_hash(hash1_str)
        h2 = imagehash.hex_to_hash(hash2_str)
        return h1 - h2
    except Exception:
        # Return maximum distance on error
        return 64

def cluster_duplicates(
    files_meta: List[Dict[str, Any]], 
    max_time_diff: float = 15.0, 
    max_hash_dist: int = 12
) -> Dict[str, str]:
    """Group similar files based on capture time and pHash.
    
    Arguments:
        files_meta: List of dicts in format:
            [{"file_path": "...", "datetime": "...", "phash": "..."}]
            
    Returns:
        Dictionary of {file_path: group_id} for duplicate files.
    """
    # 1. Filter files with valid datetime and pHash, sort by timestamp
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
    # Build adjacency graph
    adj: Dict[str, Set[str]] = {f["file_path"]: set() for f in valid_files}
    
    # 2. Pairwise comparison in a sliding time window
    for i in range(n):
        f1 = valid_files[i]
        path1 = f1["file_path"]
        t1 = f1["timestamp"]
        h1 = f1["phash"]
        
        # Compare with subsequent files while time difference <= max_time_diff
        for j in range(i + 1, n):
            f2 = valid_files[j]
            t2 = f2["timestamp"]
            
            if t2 - t1 > max_time_diff:
                break
                
            path2 = f2["file_path"]
            h2 = f2["phash"]
            
            # Calculate Hamming distance
            dist = get_hamming_distance(h1, h2)
            if dist <= max_hash_dist:
                adj[path1].add(path2)
                adj[path2].add(path1)
                
    # 3. Find connected components (BFS traversal)
    visited: Set[str] = set()
    groups: List[List[str]] = []
    
    for f in valid_files:
        path = f["file_path"]
        if path not in visited:
            # Start component traversal
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
            
            # If component has more than one file, it is a duplicate series
            if len(component) >= 2:
                groups.append(component)
                
    # 4. Construct result dictionary {path: group_id}
    file_to_group = {}
    for idx, group in enumerate(groups):
        group_id = f"dup_group_{idx + 1:03d}"
        for path in group:
            file_to_group[path] = group_id
            
    return file_to_group

