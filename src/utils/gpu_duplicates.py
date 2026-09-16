"""
GPU-Accelerated & Vectorized Duplicate / Similarity Detection Engine
Offloads batch perceptual hashing (dHash) and pairwise tensor Hamming distance
matrices to NVIDIA GPU (CUDA) via PyTorch, with automatic CPU fallback.
"""

import os
import time
import math
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from PIL import Image
import numpy as np

try:
    import torch
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
except ImportError:
    torch = None
    TF = None
    HAS_TORCH = False


def is_cuda_available() -> bool:
    if HAS_TORCH and torch.cuda.is_available():
        return True
    return False


def get_gpu_device_info() -> Dict[str, Any]:
    if is_cuda_available():
        return {
            "available": True,
            "engine": "gpu",
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024, 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2),
        }
    return {
        "available": False,
        "engine": "cpu",
        "device_name": "CPU Vectorized Mode",
    }


def parse_timestamp(dt_str: Optional[str], mtime: Optional[float] = None) -> float:
    if dt_str:
        dt_str = str(dt_str).strip()
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                clean_str = dt_str[:19].replace("-", ":").replace("T", " ")
                dt = datetime.strptime(clean_str, "%Y:%m:%d %H:%M:%S")
                return dt.timestamp()
            except ValueError:
                continue
    return mtime or 0.0


def compute_content_hash(file_path: str, file_size: int) -> str:
    """Fast sparse/full content hash to identify 100% exact duplicates without freezing disk I/O."""
    try:
        h = hashlib.md5()
        if file_size <= 20 * 1024 * 1024:
            with open(file_path, "rb") as f:
                while chunk := f.read(256 * 1024):
                    h.update(chunk)
            return h.hexdigest()
        
        # Sparse chunk hashing for large files (>20MB)
        h.update(f"size:{file_size}".encode("utf-8"))
        with open(file_path, "rb") as f:
            # Head
            h.update(f.read(64 * 1024))
            # Mid
            f.seek(file_size // 2)
            h.update(f.read(64 * 1024))
            # Tail
            f.seek(max(0, file_size - 64 * 1024))
            h.update(f.read(64 * 1024))
        return h.hexdigest()
    except Exception:
        return ""


def compute_dhash_pil(image_path: str) -> Optional[Tuple[str, List[int]]]:
    """Calculate 64-bit dHash using PIL/NumPy (CPU fallback). Returns (hex_str, 64_bits_list)."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.int16)
            diff = (arr[:, :-1] > arr[:, 1:]).astype(np.uint8).flatten()
            bits = diff.tolist()
            
            # Convert 64 bits to 16-hex string
            hex_str = "".join(
                format(int("".join(map(str, bits[i:i+4])), 2), "x")
                for i in range(0, 64, 4)
            )
            return hex_str, bits
    except Exception:
        return None


def batch_compute_dhash_gpu(image_paths: List[str], batch_size: int = 64) -> Dict[str, Tuple[str, List[int]]]:
    """Compute 64-bit dHash across batch of images using GPU tensor operations if PyTorch CUDA is active."""
    results: Dict[str, Tuple[str, List[int]]] = {}
    use_gpu = is_cuda_available()
    device = torch.device("cuda:0") if use_gpu else (torch.device("cpu") if HAS_TORCH else None)

    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        tensors = []
        valid_chunk_paths = []

        for p in chunk:
            try:
                with Image.open(p) as img:
                    img_gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
                    if HAS_TORCH and TF is not None:
                        t = TF.to_tensor(img_gray).squeeze(0)  # Shape: (8, 9)
                        tensors.append(t)
                        valid_chunk_paths.append(p)
                    else:
                        pil_res = compute_dhash_pil(p)
                        if pil_res:
                            results[p] = pil_res
            except Exception:
                continue

        if HAS_TORCH and tensors and device:
            try:
                batch_tensor = torch.stack(tensors).to(device)  # (B, 8, 9)
                left = batch_tensor[:, :, :-1]  # (B, 8, 8)
                right = batch_tensor[:, :, 1:]   # (B, 8, 8)
                diff = (left > right).flatten(start_dim=1).to(torch.uint8)  # (B, 64)

                diff_cpu = diff.cpu().numpy()
                for idx, path_str in enumerate(valid_chunk_paths):
                    bits = diff_cpu[idx].tolist()
                    hex_str = "".join(
                        format(int("".join(map(str, bits[k:k+4])), 2), "x")
                        for k in range(0, 64, 4)
                    )
                    results[path_str] = (hex_str, bits)
            except Exception:
                # Fallback to CPU per-item
                for p in valid_chunk_paths:
                    pil_res = compute_dhash_pil(p)
                    if pil_res:
                        results[p] = pil_res

    return results


def rank_group_members(members: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
    """Non-destructive ranking heuristic to designate the recommended primary photo in a group."""
    def sort_key(item):
        w = item.get("width") or 0
        h = item.get("height") or 0
        mp = w * h
        size = item.get("file_size") or 0
        mtime = item.get("mtime") or 0
        
        if strategy == "highest_resolution":
            return (-mp, -size, -mtime)
        elif strategy == "largest_file_size":
            return (-size, -mp, -mtime)
        elif strategy == "newest":
            return (-mtime, -mp, -size)
        elif strategy == "oldest":
            return (mtime, -mp, -size)
        return (-mp, -size)

    return sorted(members, key=sort_key)


def run_gpu_duplicate_clustering(
    files: List[Dict[str, Any]],
    mode: str = "all",
    similarity_threshold: float = 0.90,
    burst_window_seconds: float = 3.0,
    keep_strategy: str = "highest_resolution",
) -> Dict[str, Any]:
    """Execute complete high-speed GPU tensor duplicate and similarity detection."""
    start_time = time.time()
    gpu_info = get_gpu_device_info()
    use_gpu = gpu_info["available"]
    device = torch.device("cuda:0") if use_gpu else (torch.device("cpu") if HAS_TORCH else None)

    # 1. Gather file metadata & hashes
    image_paths = [f["file_path"] for f in files if f.get("is_image", True)]
    hash_cache = batch_compute_dhash_gpu(image_paths)

    enriched_files: List[Dict[str, Any]] = []
    for f in files:
        fp = f["file_path"]
        size = f.get("file_size")
        mtime = f.get("mtime")
        
        if (size is None or mtime is None) and os.path.exists(fp):
            stat = os.stat(fp)
            size = stat.st_size
            mtime = stat.st_mtime
            
        content_hash = compute_content_hash(fp, size or 0)
        phash_info = hash_cache.get(fp)
        phash_str = phash_info[0] if phash_info else None
        phash_bits = phash_info[1] if phash_info else None

        w, h = f.get("width"), f.get("height")
        if (not w or not h) and os.path.exists(fp):
            try:
                with Image.open(fp) as img:
                    w, h = img.size
            except Exception:
                w, h = None, None

        mp = round((w * h) / 1000000, 2) if (w and h) else None

        enriched_files.append({
            "file_path": fp,
            "filename": os.path.basename(fp),
            "folder": os.path.dirname(fp),
            "file_size": size or 0,
            "mtime": mtime or 0,
            "width": w,
            "height": h,
            "megapixels": mp,
            "content_hash": content_hash,
            "phash": phash_str,
            "_bits": phash_bits,
            "is_image": f.get("is_image", True),
            "is_video": f.get("is_video", False),
        })

    groups: List[Dict[str, Any]] = []
    assigned_files: Set[str] = set()

    # 2. Exact Duplicate Clustering
    if mode in ("all", "exact"):
        content_map: Dict[str, List[Dict[str, Any]]] = {}
        for ef in enriched_files:
            ch = ef.get("content_hash")
            if ch:
                content_map.setdefault(ch, []).append(ef)

        for ch, members in content_map.items():
            if len(members) > 1:
                sorted_members = rank_group_members(members, keep_strategy)
                primary = dict(sorted_members[0])
                primary["is_primary"] = True
                
                duplicates = []
                for d in sorted_members[1:]:
                    d_copy = dict(d)
                    d_copy["is_primary"] = False
                    d_copy["similarity_to_primary"] = 1.0
                    duplicates.append(d_copy)
                    assigned_files.add(d["file_path"])
                assigned_files.add(primary["file_path"])

                reclaimable = sum(d["file_size"] for d in duplicates)
                folders = list(set(m["folder"] for m in sorted_members))

                groups.append({
                    "id": f"exact_{ch[:12]}",
                    "match_type": "exact",
                    "similarity": 1.0,
                    "primary_file": primary,
                    "duplicates": duplicates,
                    "total_files": len(sorted_members),
                    "reclaimable_bytes": reclaimable,
                    "folder_breakdown": folders,
                })

    # 3. Vectorized GPU / CPU Visual Similarity Clustering
    if mode in ("all", "visual"):
        remaining_images = [
            ef for ef in enriched_files
            if ef["file_path"] not in assigned_files and ef.get("_bits") and len(ef["_bits"]) == 64
        ]

        if len(remaining_images) > 1:
            max_dist = int(math.floor((1.0 - similarity_threshold) * 64))
            
            n = len(remaining_images)
            dist_matrix = np.zeros((n, n), dtype=np.uint8)
            CHUNK_SIZE = 512

            if HAS_TORCH and device and use_gpu:
                # Vectorized Matrix Multiplication on GPU in bounded memory chunks
                bits_matrix = torch.tensor([ef["_bits"] for ef in remaining_images], dtype=torch.uint8, device=device)  # (N, 64)
                for start_idx in range(0, n, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n)
                    chunk_bits = bits_matrix[start_idx:end_idx]
                    chunk_dist = (chunk_bits.unsqueeze(1) ^ bits_matrix.unsqueeze(0)).sum(dim=-1)
                    dist_matrix[start_idx:end_idx] = chunk_dist.cpu().numpy()
            elif HAS_TORCH and device:
                # Vectorized PyTorch CPU in bounded memory chunks
                bits_matrix = torch.tensor([ef["_bits"] for ef in remaining_images], dtype=torch.uint8, device=device)
                for start_idx in range(0, n, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n)
                    chunk_bits = bits_matrix[start_idx:end_idx]
                    chunk_dist = (chunk_bits.unsqueeze(1) ^ bits_matrix.unsqueeze(0)).sum(dim=-1)
                    dist_matrix[start_idx:end_idx] = chunk_dist.numpy()
            else:
                # Vectorized NumPy CPU fallback in bounded memory chunks
                bits_matrix = np.array([ef["_bits"] for ef in remaining_images], dtype=np.uint8)
                for start_idx in range(0, n, CHUNK_SIZE):
                    end_idx = min(start_idx + CHUNK_SIZE, n)
                    chunk_bits = bits_matrix[start_idx:end_idx]
                    chunk_dist = np.bitwise_xor(chunk_bits[:, np.newaxis, :], bits_matrix[np.newaxis, :, :]).sum(axis=-1)
                    dist_matrix[start_idx:end_idx] = chunk_dist
            for i in range(n):
                file_a = remaining_images[i]
                if file_a["file_path"] in assigned_files:
                    continue
                
                cluster = [file_a]
                min_sim = 1.0
                for j in range(i + 1, n):
                    file_b = remaining_images[j]
                    if file_b["file_path"] in assigned_files:
                        continue
                    dist = int(dist_matrix[i, j])
                    if dist <= max_dist:
                        sim = round(1.0 - dist / 64.0, 3)
                        min_sim = min(min_sim, sim)
                        cluster.append(file_b)

                if len(cluster) > 1:
                    sorted_members = rank_group_members(cluster, keep_strategy)
                    primary = dict(sorted_members[0])
                    primary["is_primary"] = True
                    
                    duplicates = []
                    for d in sorted_members[1:]:
                        d_copy = dict(d)
                        d_copy["is_primary"] = False
                        d_copy["similarity_to_primary"] = min_sim
                        duplicates.append(d_copy)
                        assigned_files.add(d["file_path"])
                    assigned_files.add(primary["file_path"])

                    reclaimable = sum(d["file_size"] for d in duplicates)
                    folders = list(set(m["folder"] for m in sorted_members))

                    groups.append({
                        "id": f"visual_{primary['phash'][:10]}_{i}",
                        "match_type": "visual",
                        "similarity": min_sim,
                        "primary_file": primary,
                        "duplicates": duplicates,
                        "total_files": len(sorted_members),
                        "reclaimable_bytes": reclaimable,
                        "folder_breakdown": folders,
                    })

    # 4. Burst Photo Series Clustering (within time delta in same folder)
    if mode in ("all", "burst"):
        remaining = [
            ef for ef in enriched_files
            if ef["file_path"] not in assigned_files and ef.get("_bits") and len(ef["_bits"]) == 64 and ef.get("mtime", 0) > 0
        ]
        remaining.sort(key=lambda x: x["mtime"])

        current_burst: List[Dict[str, Any]] = []
        req_sim = similarity_threshold if similarity_threshold is not None else 0.85
        max_burst_dist = int(math.floor((1.0 - req_sim) * 64))

        for item in remaining:
            if item["file_path"] in assigned_files:
                continue

            if not current_burst:
                current_burst.append(item)
                continue

            anchor = current_burst[0]
            prev = current_burst[-1]
            time_diff_anchor = abs(item["mtime"] - anchor["mtime"])
            time_diff_prev = abs(item["mtime"] - prev["mtime"])
            is_time_match = time_diff_anchor <= burst_window_seconds and time_diff_prev <= burst_window_seconds
            is_folder_match = item["folder"] == anchor["folder"]

            is_visual_match = False
            if item.get("_bits") and anchor.get("_bits"):
                dist = sum(b1 ^ b2 for b1, b2 in zip(item["_bits"], anchor["_bits"]))
                is_visual_match = dist <= max_burst_dist

            if is_time_match and is_folder_match and is_visual_match:
                current_burst.append(item)
            else:
                if len(current_burst) > 1:
                    _push_burst(current_burst, groups, assigned_files, keep_strategy)
                current_burst = [item]

        if len(current_burst) > 1:
            _push_burst(current_burst, groups, assigned_files, keep_strategy)

    # Clean temporary internal fields
    for g in groups:
        g["primary_file"].pop("_bits", None)
        for d in g["duplicates"]:
            d.pop("_bits", None)

    groups.sort(key=lambda x: x["reclaimable_bytes"], reverse=True)
    elapsed = round(time.time() - start_time, 3)

    return {
        "engine": gpu_info["engine"],
        "device_name": gpu_info["device_name"],
        "elapsed_seconds": elapsed,
        "scanned_files_count": len(files),
        "total_groups": len(groups),
        "groups": groups,
    }


def _push_burst(
    burst: List[Dict[str, Any]],
    groups: List[Dict[str, Any]],
    assigned_files: Set[str],
    keep_strategy: str,
):
    sorted_members = rank_group_members(burst, keep_strategy)
    primary = dict(sorted_members[0])
    primary["is_primary"] = True
    
    min_sim = 1.0
    duplicates = []
    for d in sorted_members[1:]:
        d_copy = dict(d)
        d_copy["is_primary"] = False
        if primary.get("_bits") and d.get("_bits"):
            dist = sum(b1 ^ b2 for b1, b2 in zip(primary["_bits"], d["_bits"]))
            sim = round(1.0 - dist / 64.0, 2)
        else:
            sim = 0.85
        min_sim = min(min_sim, sim)
        d_copy["similarity_to_primary"] = sim
        duplicates.append(d_copy)
        assigned_files.add(d["file_path"])
    assigned_files.add(primary["file_path"])

    reclaimable = sum(d["file_size"] for d in duplicates)
    folders = list(set(m["folder"] for m in sorted_members))

    groups.append({
        "id": f"burst_{os.path.basename(primary['file_path']).split('.')[0]}",
        "match_type": "burst",
        "similarity": round(min_sim, 2),
        "primary_file": primary,
        "duplicates": duplicates,
        "total_files": len(sorted_members),
        "reclaimable_bytes": reclaimable,
        "folder_breakdown": folders,
    })
