import os
import urllib.request
import urllib.parse
import tempfile
from pathlib import Path
from typing import Optional, Union
from src import config

def is_running_in_docker() -> bool:
    """Detect if current process is running inside a Docker container."""
    return os.path.isfile("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER", "").lower() in ("true", "1")

def get_filename_from_path(file_path: Union[str, Path]) -> str:
    """Extract filename correctly from Windows UNC, Windows drive, or POSIX paths."""
    path_str = str(file_path).replace("\\", "/").rstrip("/")
    return path_str.split("/")[-1] if "/" in path_str else path_str

def resolve_or_stream_media_file(
    file_path: Union[str, Path],
    stream_url: Optional[str] = None,
    ui_base_url: Optional[str] = None,
    timeout: int = 15,
    local_path: Optional[Union[str, Path]] = None,
    content_base64: Optional[str] = None
) -> str:
    """
    Returns a local filepath ready for OpenCV, Pillow, Whisper, or Gemini processing.
    1. If local_path is provided and exists, returns local_path directly (e.g. from API upload).
    2. If content_base64 is provided, decodes to temporary file and returns it.
    3. If file_path is accessible locally, returns file_path as a string.
    4. Otherwise, check container directory (/app/media_input) or configured INPUT_FOLDERS by filename.
    5. If not found locally, streams media bytes from stream_url or UI API to a temporary local file.
    6. If none works, raises FileNotFoundError with a descriptive message.
    """
    # 0. Check pre-loaded / uploaded local path from API
    if local_path:
        local_path_str = str(local_path)
        if os.path.isfile(local_path_str) and os.access(local_path_str, os.R_OK):
            return local_path_str

    # 0b. Check base64 encoded media payload
    if content_base64:
        import base64
        b64_str = str(content_base64).strip()
        if "," in b64_str and ";base64" in b64_str[:60]:
            b64_str = b64_str.split(",", 1)[1]
        decoded = base64.b64decode(b64_str)
        fname = get_filename_from_path(file_path)
        ext = Path(fname).suffix if fname else (Path(str(file_path)).suffix or ".tmp")
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="be_media_b64_", suffix=ext)
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(decoded)
        return tmp_path

    file_path_str = str(file_path)

    # 1. Direct filesystem check
    if os.path.isfile(file_path_str) and os.access(file_path_str, os.R_OK):
        return file_path_str

    try:
        p = Path(file_path_str)
        if p.is_file() and os.access(str(p), os.R_OK):
            return str(p)
    except Exception:
        pass

    # 1.5. Check PATH_MAPPINGS to translate remote host/container path prefixes
    path_mappings = getattr(config, "PATH_MAPPINGS", {}) or {}
    norm_file_path = file_path_str.replace("\\", "/")
    for src_prefix, dst_prefix in path_mappings.items():
        norm_src = str(src_prefix).replace("\\", "/").rstrip("/")
        norm_dst = str(dst_prefix).replace("\\", "/").rstrip("/")
        if norm_file_path == norm_src or norm_file_path.startswith(norm_src + "/"):
            rel_part = norm_file_path[len(norm_src):].lstrip("/")
            candidate = f"{norm_dst}/{rel_part}" if rel_part else norm_dst
            if os.path.isfile(candidate) and os.access(candidate, os.R_OK):
                return os.path.normpath(candidate)
            try:
                cand_p = Path(candidate)
                if cand_p.is_file() and os.access(str(cand_p), os.R_OK):
                    return str(cand_p)
            except Exception:
                pass

    # 2. Check container fallback directory (/app/media_input) & configured INPUT_FOLDERS
    fname = get_filename_from_path(file_path_str)
    search_dirs = []
    if os.path.isdir("/app/media_input"):
        search_dirs.append(Path("/app/media_input"))
    for folder in getattr(config, "INPUT_FOLDERS", []):
        try:
            f_p = Path(folder)
            if f_p.is_dir() and f_p not in search_dirs:
                search_dirs.append(f_p)
        except Exception:
            pass

    if fname:
        for sdir in search_dirs:
            try:
                # Direct flat filename check
                cand = sdir / fname
                if cand.is_file() and os.access(str(cand), os.R_OK):
                    return str(cand)

                # Subpath check if file_path_str has directory segments
                clean_parts = [part for part in norm_file_path.split("/") if part and part not in ("app", "media_input", "nas", "media")]
                if len(clean_parts) > 1:
                    subpath_cand = sdir / "/".join(clean_parts)
                    if subpath_cand.is_file() and os.access(str(subpath_cand), os.R_OK):
                        return str(subpath_cand)
            except Exception:
                pass

    # 3. Determine target stream URL
    target_url = stream_url
    configured_ui_base = getattr(config, "UI_BASE_URL", None) or os.environ.get("UI_BASE_URL")
    effective_ui_base = ui_base_url or configured_ui_base or "http://localhost:8000"
    if not target_url and effective_ui_base:
        encoded_path = urllib.parse.quote(file_path_str)
        target_url = f"{effective_ui_base.rstrip('/')}/api/media/file?path={encoded_path}"

    if target_url:
        # If configured_ui_base is an external IP/domain, rewrite localhost/127.0.0.1 in incoming stream_url
        if configured_ui_base and "localhost" not in configured_ui_base and "127.0.0.1" not in configured_ui_base:
            if "localhost:" in target_url or "127.0.0.1:" in target_url:
                try:
                    p_cfg = urllib.parse.urlparse(configured_ui_base)
                    p_tgt = urllib.parse.urlparse(target_url)
                    target_url = urllib.parse.urlunparse(p_tgt._replace(netloc=p_cfg.netloc, scheme=p_cfg.scheme or "http"))
                except Exception:
                    pass

        # If running inside Docker and target is localhost/127.0.0.1, map to host.docker.internal
        if is_running_in_docker():
            if "localhost:" in target_url:
                target_url = target_url.replace("localhost:", "host.docker.internal:")
            elif "127.0.0.1:" in target_url:
                target_url = target_url.replace("127.0.0.1:", "host.docker.internal:")

        tmp_path = None
        try:
            ext = Path(fname).suffix if fname else (Path(file_path_str).suffix or ".tmp")
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="be_media_", suffix=ext)
            os.close(tmp_fd)

            req = urllib.request.Request(
                target_url,
                headers={"User-Agent": "MediaCataloger-BE/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(tmp_path, "wb") as out_f:
                    while True:
                        chunk = response.read(65536)
                        if not chunk:
                            break
                        out_f.write(chunk)

            if os.path.isfile(tmp_path) and os.path.getsize(tmp_path) > 0:
                return tmp_path
            else:
                if os.path.isfile(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            raise FileNotFoundError(f"Media file '{file_path_str}' is not accessible on BE filesystem (streaming failed from {target_url}: {e})")

    raise FileNotFoundError(
        f"Media file '{file_path_str}' is not accessible on BE filesystem. Provide the mediafile outside via API (multipart upload 'file_upload', 'content_base64', or reachable 'stream_url')."
    )
