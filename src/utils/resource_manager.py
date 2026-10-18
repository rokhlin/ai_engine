"""
Resource Manager for AI Engine
Provides on-demand model unloading, PyTorch CUDA cache clearing,
and Python garbage collection to prevent memory bloat in Docker containers.
"""

import gc
from typing import Dict, Any, List

def release_pipeline_resources() -> Dict[str, Any]:
    """
    Release all heavy ML models (faces, whisper), clear CUDA cache, and run GC.
    Called automatically upon pipeline completion or stop, and via POST /api/cleanup.
    """
    freed: List[str] = []

    # 1. Unload InsightFace analyzer
    try:
        from src.utils.faces import unload_face_analyzer
        unload_face_analyzer()
        freed.append("faces")
    except Exception as e:
        print(f"[ResourceManager] Warning unloading face analyzer: {e}")

    # 2. Unload Whisper model
    try:
        from src.utils.transcribe import unload_whisper_model
        unload_whisper_model()
        freed.append("whisper")
    except Exception as e:
        print(f"[ResourceManager] Warning unloading whisper model: {e}")

    # 3. Clear CUDA / PyTorch memory if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            freed.append("cuda_cache")
    except Exception:
        pass

    # 4. Force Python garbage collection
    collected = gc.collect()
    print(f"[ResourceManager] Pipeline resources released ({', '.join(freed)}), GC collected {collected} objects.")
    return {
        "status": "success",
        "freed": freed,
        "gc_collected": collected
    }
