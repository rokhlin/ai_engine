import pytest
import os
import tempfile
from pathlib import Path
from PIL import Image
from src.utils.gpu_duplicates import (
    get_gpu_device_info,
    compute_content_hash,
    compute_dhash_pil,
    batch_compute_dhash_gpu,
    run_gpu_duplicate_clustering,
    rank_group_members,
)


def test_gpu_device_info():
    info = get_gpu_device_info()
    assert "engine" in info
    assert "device_name" in info
    if info["available"]:
        assert info["engine"] == "gpu"
        assert info["device_count"] >= 1
    else:
        assert info["engine"] == "cpu"
        assert "CPU" in info["device_name"]


def test_gpu_device_info_cuda_mocked(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(idx):
            return "NVIDIA GeForce RTX 4080"

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def memory_allocated(idx):
            return 1024 * 1024 * 100

        @staticmethod
        def memory_reserved(idx):
            return 1024 * 1024 * 200

    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda
    monkeypatch.setattr("src.utils.gpu_duplicates.HAS_TORCH", True)
    monkeypatch.setattr("src.utils.gpu_duplicates.torch", fake_torch)

    info = get_gpu_device_info()
    assert info["engine"] == "gpu"
    assert info["available"] is True
    assert "RTX 4080" in info["device_name"]
    assert info["device_count"] == 1
    assert info["memory_allocated_mb"] == 100.0


def test_exact_content_hash_and_clustering():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.png"
        img_unique = tmp_path / "unique.png"

        im = Image.new("RGB", (100, 100), color="red")
        im.save(img1)
        im.save(img2)

        im_u = Image.new("RGB", (100, 100), color="blue")
        im_u.save(img_unique)

        files = [
            {"file_path": str(img1), "file_size": os.path.getsize(img1), "mtime": 1000, "width": 100, "height": 100},
            {"file_path": str(img2), "file_size": os.path.getsize(img2), "mtime": 2000, "width": 100, "height": 100},
            {"file_path": str(img_unique), "file_size": os.path.getsize(img_unique), "mtime": 3000, "width": 100, "height": 100},
        ]

        result = run_gpu_duplicate_clustering(files, mode="exact", keep_strategy="newest")
        assert result["total_groups"] == 1
        group = result["groups"][0]
        assert group["match_type"] == "exact"
        assert group["total_files"] == 2
        assert group["primary_file"]["file_path"] == str(img2)
        assert group["duplicates"][0]["file_path"] == str(img1)


def test_visual_similarity_clustering_gpu():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create images with distinct gradient patterns
        im_base = Image.new("RGB", (120, 120))
        for x in range(120):
            for y in range(120):
                im_base.putpixel((x, y), (x * 2, y * 2, (x + y)))
        
        img_a = tmp_path / "sim_a.jpg"
        img_b = tmp_path / "sim_b.jpg"
        img_c = tmp_path / "diff.jpg"

        im_base.save(img_a)
        
        # Near identical gradient (minor perturbation)
        im_modified = im_base.copy()
        im_modified.putpixel((10, 10), (250, 250, 250))
        im_modified.save(img_b)

        # Inverted gradient (visually opposite)
        im_diff = Image.new("RGB", (120, 120))
        for x in range(120):
            for y in range(120):
                im_diff.putpixel((x, y), (255 - x * 2, 255 - y * 2, 255 - (x + y)))
        im_diff.save(img_c)

        files = [
            {"file_path": str(img_a), "file_size": os.path.getsize(img_a), "mtime": 1000, "width": 120, "height": 120},
            {"file_path": str(img_b), "file_size": os.path.getsize(img_b) + 50, "mtime": 2000, "width": 120, "height": 120},
            {"file_path": str(img_c), "file_size": os.path.getsize(img_c), "mtime": 3000, "width": 120, "height": 120},
        ]

        result = run_gpu_duplicate_clustering(files, mode="visual", similarity_threshold=0.90, keep_strategy="largest_file_size")
        assert result["total_groups"] == 1
        group = result["groups"][0]
        assert group["match_type"] == "visual"
        assert group["total_files"] == 2
        assert group["primary_file"]["file_path"] == str(img_b)
        assert group["similarity"] >= 0.90


def test_burst_photo_series_clustering():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        b1 = tmp_path / "burst_01.jpg"
        b2 = tmp_path / "burst_02.jpg"
        b3 = tmp_path / "burst_03.jpg"

        im = Image.new("RGB", (400, 300), color="green")
        im.save(b1)
        im.save(b2)
        im.save(b3)

        files = [
            {"file_path": str(b1), "file_size": 1000, "mtime": 1000.0, "width": 400, "height": 300},
            {"file_path": str(b2), "file_size": 1200, "mtime": 1001.5, "width": 400, "height": 300},
            {"file_path": str(b3), "file_size": 1100, "mtime": 1003.0, "width": 400, "height": 300},
        ]

        result = run_gpu_duplicate_clustering(files, mode="burst", burst_window_seconds=3.0, keep_strategy="largest_file_size")
        assert result["total_groups"] == 1
        group = result["groups"][0]
        assert group["match_type"] == "burst"
        assert group["total_files"] == 3
        assert group["primary_file"]["file_path"] == str(b2)


def test_ranking_heuristics():
    items = [
        {"file_path": "a.jpg", "width": 1000, "height": 1000, "file_size": 500, "mtime": 10},
        {"file_path": "b.jpg", "width": 2000, "height": 2000, "file_size": 400, "mtime": 20},
    ]

    res_ranked = rank_group_members(items, "highest_resolution")
    assert res_ranked[0]["file_path"] == "b.jpg"

    size_ranked = rank_group_members(items, "largest_file_size")
    assert size_ranked[0]["file_path"] == "a.jpg"

    newest_ranked = rank_group_members(items, "newest")
    assert newest_ranked[0]["file_path"] == "b.jpg"
