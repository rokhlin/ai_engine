import os
import sys
import json
import time
import tempfile
import hashlib
import traceback
import threading
import inspect
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import config
from src import database
from src.utils import exif, faces, video, hash, gemini
from src.utils.media_loader import resolve_or_stream_media_file

def _extract_item_info(item: Any, default_root: Optional[Path] = None):
    """Helper to extract logical attributes from Path, dict, or Pydantic item."""
    local_path = None
    content_base64 = None
    if hasattr(item, "file_path") or hasattr(item, "file"):
        file_path = str(getattr(item, "file_path", None) or getattr(item, "file", None) or "")
        folder = str(getattr(item, "folder", None) or "")
        filename = str(getattr(item, "filename", None) or "")
        file_size = int(getattr(item, "file_size", None) or 0)
        mtime = float(getattr(item, "mtime", None) or 0.0)
        stream_url = getattr(item, "stream_url", None)
        local_path = getattr(item, "local_path", None)
        content_base64 = getattr(item, "content_base64", None)
    elif isinstance(item, dict):
        file_path = str(item.get("file_path") or item.get("file") or "")
        folder = str(item.get("folder") or "")
        filename = str(item.get("filename") or "")
        file_size = int(item.get("file_size") or 0)
        mtime = float(item.get("mtime") or 0.0)
        stream_url = item.get("stream_url")
        local_path = item.get("local_path")
        content_base64 = item.get("content_base64")
    elif isinstance(item, tuple) and len(item) >= 2:
        file_path = str(item[0])
        folder = str(item[1])
        filename = Path(file_path.replace("\\", "/")).name
        file_size = 0
        mtime = 0.0
        stream_url = None
    else:
        file_path = str(item)
        folder = str(default_root) if default_root else ""
        filename = Path(file_path.replace("\\", "/")).name
        file_size = 0
        mtime = 0.0
        stream_url = None

    if not filename and file_path:
        filename = Path(file_path.replace("\\", "/")).name

    return file_path, folder, filename, file_size, mtime, stream_url, local_path, content_base64

ALL_SUPPORTED_MODES = {"transcribe", "faces", "duplicates", "vision"}

def normalize_modes(modes: Optional[Union[str, List[str], set]]) -> set:
    """Normalize input mode(s) to a standardized set of active modules."""
    if not modes or modes == "all":
        return set(ALL_SUPPORTED_MODES)
    if isinstance(modes, str):
        parts = {p.strip().lower() for p in modes.replace(";", ",").split(",") if p.strip()}
        if "all" in parts or not parts:
            return set(ALL_SUPPORTED_MODES)
        return parts.intersection(ALL_SUPPORTED_MODES)
    if isinstance(modes, (list, tuple, set)):
        items = {str(m).strip().lower() for m in modes if m}
        if "all" in items or not items:
            return set(ALL_SUPPORTED_MODES)
        return items.intersection(ALL_SUPPORTED_MODES)
    return set(ALL_SUPPORTED_MODES)

def get_file_modules_status(sidecar_data: Dict[str, Any], media_type: str = "photo") -> Dict[str, Any]:
    """Calculate completion status of each AI module for a media item."""
    transcription = sidecar_data.get("transcription") or (sidecar_data.get("gemini_analysis", {}) or {}).get("transcription")
    has_transcribe = bool(transcription) if media_type == "video" else None
    
    if media_type == "photo":
        has_faces = "faces" in sidecar_data and isinstance(sidecar_data["faces"], list)
    else:
        has_faces = "face_timeline" in sidecar_data and isinstance(sidecar_data["face_timeline"], dict)
        
    has_duplicates = bool(sidecar_data.get("phash"))
    has_vision = bool(sidecar_data.get("gemini_analysis") and not (sidecar_data.get("gemini_analysis", {}) or {}).get("error"))
    
    return {
        "transcribe": has_transcribe,
        "faces": has_faces,
        "duplicates": has_duplicates,
        "vision": has_vision
    }




def is_vault_path(path: Union[str, Path]) -> bool:
    """Check if a file or directory path is located inside a Secret Vault folder."""
    p_str = str(path).replace("\\", "/").lower()
    parts = p_str.split("/")
    vault_markers = {".vault", "vault", "secret_vault", "private_vault"}
    return any(part in vault_markers for part in parts)


def scan_input_folders(
    allow_vault: bool = False,
    input_folders: Optional[List[Union[str, Path]]] = None
) -> List[Tuple[Path, Path]]:
    """Scans input folders and returns a list of tuples (file_path, scan_root). Vault paths are isolated unless allow_vault=True."""
    from src.db_sqlite import _current_user_id, _current_user_root
    
    targets = []
    if input_folders:
        targets = [Path(p) for p in input_folders if str(p).strip()]
    elif _current_user_id.get():
        from src.user_workspace import get_user_root
        user_dir = get_user_root(_current_user_id.get(), _current_user_root.get())
        user_dir.mkdir(parents=True, exist_ok=True)
        targets = [user_dir]
    else:
        targets = list(config.INPUT_FOLDERS)

    media_files = []
    for folder in targets:
        scan_target = folder
        if not scan_target.exists():
            cand1 = Path("/app/media_input")
            cand2 = config.PROJECT_ROOT / "media_input"
            if cand1.is_dir():
                scan_target = cand1
            elif cand2.is_dir():
                scan_target = cand2

        if not scan_target.exists():
            print(f"Warning: Scan folder {folder} does not exist. Skipping.")
            continue
        
        print(f"Scanning folder recursively: {scan_target} (logical source: {folder}) ...")
        file_count = 0
        try:
            for root, dirs, files in os.walk(str(scan_target)):
                dirs[:] = [
                    d for d in dirs
                    if (not d.startswith(".") or (allow_vault and d.lower() in (".vault", ".secret_vault", ".private_vault")))
                    and (allow_vault or d.lower() not in (".vault", "vault", "secret_vault", "private_vault"))
                ]
                
                for file in files:
                    if file.startswith(".") and not (allow_vault and is_vault_path(Path(root) / file)):
                        continue
                    path = Path(root) / file
                    if not allow_vault and is_vault_path(path):
                        continue
                    ext = path.suffix.lower()
                    if ext in config.SUPPORTED_PHOTO_EXTS or ext in config.SUPPORTED_VIDEO_EXTS:
                        media_files.append((path, folder))
                        file_count += 1
                        if file_count % 100 == 0:
                            print(f"Scanning... found {file_count} files in {folder}...", end="\r", flush=True)
            print(f"Scanning completed: found {file_count} files in {folder}.")
        except Exception as e:
            print(f"Warning: Error scanning folder {folder}: {e}")
            
    return media_files

def get_relative_output_path(
    input_path: Union[Path, str],
    input_root: Optional[Union[Path, str]] = None,
    output_folder: Optional[Union[Path, str]] = None,
    preserve_structure: Optional[bool] = None
) -> Path:
    """Calculates sidecar file path in OUTPUT_FOLDER cross-platform."""
    out_dir = Path(output_folder) if output_folder else config.OUTPUT_FOLDER
    preserve = preserve_structure if preserve_structure is not None else config.PRESERVE_STRUCTURE

    clean_input_str = str(input_path).replace("\\", "/")
    filename = Path(clean_input_str).name

    if preserve and input_root:
        clean_root_str = str(input_root).replace("\\", "/").rstrip("/")
        if clean_input_str.lower().startswith(clean_root_str.lower()):
            rel_str = clean_input_str[len(clean_root_str):].lstrip("/")
            rel_path = Path(rel_str)
        else:
            rel_path = Path(filename)
        out_path = out_dir / rel_path
    else:
        out_path = out_dir / filename
        
    return out_path.with_name(f"{filename}.json")



def format_timestamp(seconds: int) -> str:
    """Format seconds into MM:SS string."""
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def group_seconds_into_intervals(seconds_list: List[int]) -> List[Tuple[str, str]]:
    """Group consecutive seconds into timecode intervals."""
    if not seconds_list:
        return []
    
    sorted_secs = sorted(list(set(seconds_list)))
    intervals = []
    
    start = sorted_secs[0]
    prev = start
    
    for s in sorted_secs[1:]:
        if s - prev <= 2:
            prev = s
        else:
            intervals.append((format_timestamp(start), format_timestamp(prev)))
            start = s
            prev = s
    intervals.append((format_timestamp(start), format_timestamp(prev)))
    return intervals


def retry_api_call(func, *args, max_retries=5, initial_delay=3, **kwargs):
    """Retry API call on network failures, rate limits, or quota limits (with fallback to local)."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            
            is_quota = (
                "quota exceeded" in err_str or 
                "exceeded your current quota" in err_str or 
                "resource_exhausted" in err_str or 
                ("429" in err_str and "quota" in err_str)
            )
            
            if is_quota:
                if config.MODEL_PROVIDER == "gemini" and config.FALLBACK_TO_LOCAL:
                    print(f"\n[WARNING] Gemini API Quota Reached! Switching to local model (LM Studio) and retrying...")
                    config.MODEL_PROVIDER = "local"
                    time.sleep(1)
                    continue
                else:
                    print(f"\n[FATAL ERROR] Gemini API Quota Reached and local fallback is not enabled/available. Stopping task.\nDetails: {e}")
                    sys.exit(1)
                
            is_rate_limit = "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str
            is_network = "timeout" in err_str or "connection" in err_str or "network" in err_str or "503" in err_str
            
            if (is_rate_limit or is_network) and attempt < max_retries - 1:
                print(f"[Retry] API call error ({e.__class__.__name__}: {e}). Attempt {attempt+1}/{max_retries}. Waiting {delay} sec...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e


# --- Media Processors ---

def process_photo(
    photo_item: Any,
    input_root: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    preserve_structure: Optional[bool] = None,
    ui_base_url: Optional[str] = None,
    target_tags: Optional[List[str]] = None,
    tag_format: Optional[str] = None,
    modes: Optional[Union[str, List[str], set]] = None,
    custom_prompt: Optional[str] = None,
    force: bool = False
) -> Optional[Dict[str, Any]]:
    """Process single photo: EXIF, Faces, pHash, Gemini / Local model, with selective modes and soft merging."""
    file_path, folder, filename, file_size, mtime, stream_url, local_path, content_base64 = _extract_item_info(photo_item, input_root)
    root_path = Path(folder) if folder else input_root
    sidecar_path = get_relative_output_path(
        Path(file_path),
        root_path,
        output_folder=output_folder,
        preserve_structure=preserve_structure
    )

    active_modes = normalize_modes(modes)
    local_path_str = None
    is_temp_download = False

    try:
        # Load existing sidecar metadata if available for soft merging
        existing_metadata: Dict[str, Any] = {}
        if sidecar_path.is_file():
            try:
                with open(sidecar_path, "r", encoding="utf-8") as sc_f:
                    existing_metadata = json.load(sc_f)
            except Exception:
                existing_metadata = {}

        # Resolve accessible local path or stream from UI / external API upload
        local_path_str = resolve_or_stream_media_file(
            file_path,
            stream_url=stream_url,
            ui_base_url=ui_base_url or config.UI_BASE_URL,
            local_path=local_path,
            content_base64=content_base64
        )

        local_path = Path(local_path_str)
        is_temp_download = (os.path.abspath(local_path_str) != os.path.abspath(file_path))

        # Retrieve file_size and mtime if not populated
        if (not file_size or not mtime) and local_path.is_file():
            stat = local_path.stat()
            if not file_size:
                file_size = stat.st_size
            if not mtime:
                mtime = stat.st_mtime

        # Fast Skip Logic
        if not force:
            sync_rec = database.get_sync_record(file_path)
            if sync_rec and sync_rec.get("status") == "PROCESSED":
                if sync_rec.get("file_size") == file_size and sync_rec.get("mtime") == mtime:
                    print(f"Skipping {file_path} - unchanged since last processing (Fast Skip).")
                    if is_temp_download:
                        try:
                            os.remove(local_path)
                        except Exception:
                            pass
                    return existing_metadata

        # 1. Mark in DB as PENDING
        database.update_sync_record(
            file_path=file_path,
            file_size=file_size,
            mtime=mtime,
            status="PENDING"
        )

        print(f"\n--- Processing photo ({','.join(active_modes)}): {filename} ({file_path}) ---")

        # 2. Local EXIF (read if missing or full run)
        if "exif" in existing_metadata and existing_metadata["exif"]:
            exif_meta = existing_metadata["exif"]
        else:
            exif_meta = exif.read_photo_metadata(local_path)

        # 3. Face detection and recognition
        if "faces" in active_modes:
            face_detections = faces.detect_faces(local_path)
            faces_with_ids = []
            for face in face_detections:
                face_id = faces.match_or_register_face(
                    embedding=face["embedding"],
                    confidence=face.get("confidence", 1.0),
                    image_path=local_path,
                    bbox=face.get("bbox"),
                    source_file=file_path
                )
                mapping = database.get_face_name_mapping()
                name = mapping.get(face_id, face_id)
                faces_with_ids.append({
                    "bbox": face["bbox"],
                    "face_id": face_id,
                    "name": name,
                    "confidence": face["confidence"]
                })
        else:
            faces_with_ids = existing_metadata.get("faces", [])

        # 4. Local pHash
        if "duplicates" in active_modes or not existing_metadata.get("phash"):
            phash = hash.calculate_phash(local_path)
        else:
            phash = existing_metadata.get("phash")

        # 5. Semantic analysis via Gemini / Local
        if "vision" in active_modes:
            effective_target_tags = target_tags if target_tags is not None else getattr(config, "TARGET_TAGS", None)
            effective_tag_format = tag_format or getattr(config, "TAG_FORMAT", "categorized")

            extra_kwargs = {}
            try:
                sig = inspect.signature(gemini.analyze_photo)
                if "target_tags" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    extra_kwargs["target_tags"] = effective_target_tags
                if "tag_format" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    extra_kwargs["tag_format"] = effective_tag_format
                if "faces_data" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    extra_kwargs["faces_data"] = faces_with_ids
                if "custom_prompt" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    extra_kwargs["custom_prompt"] = custom_prompt or getattr(config, "VISION_PROMPT_TEMPLATE", None)
            except Exception:
                pass

            gemini_res = retry_api_call(
                gemini.analyze_photo, 
                local_path, 
                exif_data=exif_meta,
                **extra_kwargs
            )
            gemini_dict = gemini_res.model_dump() if hasattr(gemini_res, "model_dump") else gemini_res

            # Format and serialize tags
            raw_tags = gemini_dict.get("tags") or []
            serialized_tags = []
            for t in raw_tags:
                if isinstance(t, dict):
                    serialized_tags.append(t)
                elif hasattr(t, "model_dump"):
                    serialized_tags.append(t.model_dump())
                elif isinstance(t, str):
                    serialized_tags.append({"tag": t, "category": "general", "confidence": 1.0})

            # Guarantee that recognized named individuals are included in tags
            existing_people_tags = {
                (t.get("tag") or "").strip().lower()
                for t in serialized_tags if isinstance(t, dict)
            }
            for f in faces_with_ids:
                if isinstance(f, dict):
                    name = (f.get("name") or "").strip()
                    fid = (f.get("face_id") or "").strip()
                    if name and name.lower() not in ("unknown", "unnamed") and not name.lower().startswith(("face_", "face-", "person_")):
                        if name.lower() not in existing_people_tags:
                            serialized_tags.append({
                                "tag": name,
                                "category": "people",
                                "confidence": 1.0
                            })
                            existing_people_tags.add(name.lower())
                    elif name and name != fid and not fid.startswith(("manual_", "face_manual_")):
                        if name.lower() not in existing_people_tags:
                            serialized_tags.append({
                                "tag": name,
                                "category": "people",
                                "confidence": 1.0
                            })
                            existing_people_tags.add(name.lower())
        else:
            gemini_dict = existing_metadata.get("gemini_analysis", {})
            serialized_tags = existing_metadata.get("tags", [])

        # 6. Metadata assembly with soft merging
        metadata = dict(existing_metadata)
        metadata.update({
            "file_path": file_path,
            "file_name": filename,
            "file_size": file_size,
            "mtime": mtime,
            "media_type": "photo",
            "content_type": gemini_dict.get("content_type") or metadata.get("content_type", "other"),
            "tags": serialized_tags,
            "exif": exif_meta,
            "phash": phash,
            "faces": faces_with_ids,
            "gemini_analysis": gemini_dict,
            "duplicate_analysis": existing_metadata.get("duplicate_analysis"),
            "modules_status": get_file_modules_status({
                "transcription": None,
                "faces": faces_with_ids,
                "phash": phash,
                "gemini_analysis": gemini_dict,
                "tags": serialized_tags
            }, "photo")
        })

        # 7. Save sidecar file
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        # 8. Ingest into relational SQLite tables
        try:
            media_id = hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]
            database.upsert_media_item(
                media_id=media_id,
                file_path=file_path,
                file_name=filename,
                media_type="photo",
                file_size=file_size,
                mtime=mtime,
                media_date=exif_meta.get("datetime") if exif_meta else None,
                phash=phash,
                status="PROCESSED"
            )
            database.upsert_media_metadata(
                media_id=media_id,
                summary=gemini_dict.get("summary") or metadata.get("summary"),
                summary_ru=gemini_dict.get("summary_ru") or metadata.get("summary_ru"),
                description=gemini_dict.get("description") or metadata.get("description"),
                description_ru=gemini_dict.get("description_ru") or metadata.get("description_ru"),
                environment=gemini_dict.get("environment"),
                lighting=gemini_dict.get("lighting"),
                weather=gemini_dict.get("weather"),
                time_of_day=gemini_dict.get("time_of_day"),
                ocr_text=gemini_dict.get("ocr_text"),
                camera_make=exif_meta.get("camera_make") if exif_meta else None,
                camera_model=exif_meta.get("camera_model") if exif_meta else None,
                latitude=exif_meta.get("latitude") if exif_meta else None,
                longitude=exif_meta.get("longitude") if exif_meta else None,
                location_name=gemini_dict.get("location_name"),
                raw_exif=exif_meta,
                raw_gemini=gemini_dict if "error" not in gemini_dict else None
            )
            if faces_with_ids:
                database.save_media_faces(media_id, faces_with_ids)
            if serialized_tags:
                database.save_media_tags(media_id, serialized_tags)
        except Exception as db_err:
            print(f"Warning: Failed to upsert relational media item for {filename}: {db_err}")

        # 9. Successful sync record
        database.update_sync_record(
            file_path=file_path,
            file_size=file_size,
            mtime=mtime,
            status="PROCESSED",
            sidecar_path=str(sidecar_path)
        )
        print(f"Successfully processed: {filename} -> {sidecar_path.name}")
        return metadata

    except Exception as e:
        print(f"Error processing photo {filename or file_path}: {e}")
        traceback.print_exc()
        database.update_sync_record(
            file_path=file_path,
            file_size=file_size,
            mtime=mtime,
            status="FAILED",
            error_message=str(e)
        )
        return None
    finally:
        if is_temp_download and local_path_str and os.path.isfile(local_path_str):
            try:
                os.unlink(local_path_str)
            except Exception:
                pass


def process_video(
    video_item: Any,
    input_root: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    preserve_structure: Optional[bool] = None,
    ui_base_url: Optional[str] = None,
    target_tags: Optional[List[str]] = None,
    tag_format: Optional[str] = None,
    modes: Optional[Union[str, List[str], set]] = None,
    custom_prompt: Optional[str] = None,
    force: bool = False
) -> Optional[Dict[str, Any]]:
    """Process video: Metadata, Transcribe, Sample frames, Faces, Gemini / Local Model."""
    file_path, folder, filename, file_size, mtime, stream_url, local_path, content_base64 = _extract_item_info(video_item, input_root)
    root_path = Path(folder) if folder else input_root
    sidecar_path = get_relative_output_path(
        Path(file_path),
        root_path,
        output_folder=output_folder,
        preserve_structure=preserve_structure
    )

    active_modes = normalize_modes(modes)
    local_path_str = None
    is_temp_download = False

    try:
        # Load existing sidecar metadata if available for soft merging
        existing_metadata: Dict[str, Any] = {}
        if sidecar_path.is_file():
            try:
                with open(sidecar_path, "r", encoding="utf-8") as sc_f:
                    existing_metadata = json.load(sc_f)
            except Exception:
                existing_metadata = {}

        # Resolve accessible local path or stream from UI / external API upload
        local_path_str = resolve_or_stream_media_file(
            file_path,
            stream_url=stream_url,
            ui_base_url=ui_base_url or config.UI_BASE_URL,
            local_path=local_path,
            content_base64=content_base64
        )

        local_path = Path(local_path_str)
        is_temp_download = (os.path.abspath(local_path_str) != os.path.abspath(file_path))

        if (not file_size or not mtime) and local_path.is_file():
            stat = local_path.stat()
            if not file_size:
                file_size = stat.st_size
            if not mtime:
                mtime = stat.st_mtime

        # Fast Skip Logic
        if not force:
            sync_rec = database.get_sync_record(file_path)
            if sync_rec and sync_rec.get("status") == "PROCESSED":
                if sync_rec.get("file_size") == file_size and sync_rec.get("mtime") == mtime:
                    print(f"Skipping {file_path} - unchanged since last processing (Fast Skip).")
                    if is_temp_download:
                        try:
                            os.remove(local_path)
                        except Exception:
                            pass
                    return existing_metadata

        # 1. Mark in DB as PENDING
        database.update_sync_record(
            file_path=file_path,
            file_size=file_size,
            mtime=mtime,
            status="PENDING"
        )

        print(f"\n--- Processing video ({','.join(active_modes)}): {filename} ({file_path}) ---")

        # 2. Local container metadata
        if "video_metadata" in existing_metadata and existing_metadata["video_metadata"]:
            video_meta = existing_metadata["video_metadata"]
        else:
            video_meta = video.read_video_metadata(local_path)

        # 3. Audio transcription (Whisper)
        transcription = existing_metadata.get("transcription") or (existing_metadata.get("gemini_analysis", {}) or {}).get("transcription")
        transcription_ru = existing_metadata.get("transcription_ru") or (existing_metadata.get("gemini_analysis", {}) or {}).get("transcription_ru")

        if "transcribe" in active_modes:
            print("Running voice recognition (Speech to Text)...")
            try:
                from src.utils.transcribe import transcribe_video_audio
                new_t, new_t_ru = transcribe_video_audio(local_path)
                if new_t:
                    transcription = new_t
                if new_t_ru:
                    transcription_ru = new_t_ru
            except Exception as e:
                print(f"Warning: Audio transcription failed: {e}")

        # 4. Face tracking
        face_intervals = existing_metadata.get("face_timeline", {})
        frames: List[Path] = []

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            if "faces" in active_modes:
                frames_dir = temp_dir / "frames"
                frames = video.extract_frames_at_1fps(local_path, frames_dir)

                face_detections_by_second: Dict[str, List[int]] = {}
                print(f"Starting face detection on {len(frames)} frames...")
                for idx, frame_path in enumerate(frames):
                    second = idx
                    detected = faces.detect_faces(frame_path)
                    for f in detected:
                        face_id = faces.match_or_register_face(
                            embedding=f["embedding"],
                            confidence=f.get("confidence", 1.0),
                            image_path=frame_path,
                            bbox=f.get("bbox"),
                            source_file=file_path
                        )
                        if face_id not in face_detections_by_second:
                            face_detections_by_second[face_id] = []
                        face_detections_by_second[face_id].append(second)

                face_intervals = {}
                mapping = database.get_face_name_mapping()
                for face_id, seconds in face_detections_by_second.items():
                    intervals = video.group_seconds_into_intervals(seconds)
                    name = mapping.get(face_id, face_id)
                    face_intervals[face_id] = {
                        "intervals": intervals,
                        "name": name
                    }

            # 5. Vision / Semantic analysis via Gemini API or Local model
            gemini_dict = existing_metadata.get("gemini_analysis", {})
            serialized_tags = existing_metadata.get("tags", [])

            if "vision" in active_modes:
                compressed_path = temp_dir / f"compressed_{local_path.name}"
                video.compress_video_for_cloud(local_path, compressed_path)

                if not frames:
                    frames_dir = temp_dir / "frames_vision"
                    frames = video.extract_frames_at_1fps(local_path, frames_dir)

                effective_target_tags = target_tags if target_tags is not None else getattr(config, "TARGET_TAGS", None)
                effective_tag_format = tag_format or getattr(config, "TAG_FORMAT", "categorized")

                extra_kwargs = {}
                try:
                    sig = inspect.signature(gemini.analyze_video)
                    if "target_tags" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        extra_kwargs["target_tags"] = effective_target_tags
                    if "tag_format" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        extra_kwargs["tag_format"] = effective_tag_format
                    if "faces_data" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        extra_kwargs["faces_data"] = face_intervals
                    if "custom_prompt" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        extra_kwargs["custom_prompt"] = custom_prompt or getattr(config, "VISION_PROMPT_TEMPLATE", None)
                except Exception:
                    pass

                print("Running Video recognition analysis...")
                gemini_res = retry_api_call(
                    gemini.analyze_video,
                    compressed_path,
                    frames=frames,
                    transcription=transcription,
                    transcription_ru=transcription_ru,
                    **extra_kwargs
                )
                gemini_dict = gemini_res.model_dump() if hasattr(gemini_res, "model_dump") else gemini_res

                # Format and serialize tags
                raw_tags = gemini_dict.get("tags") or []
                serialized_tags = []
                for t in raw_tags:
                    if isinstance(t, dict):
                        serialized_tags.append(t)
                    elif hasattr(t, "model_dump"):
                        serialized_tags.append(t.model_dump())
                    elif isinstance(t, str):
                        serialized_tags.append({"tag": t, "category": "general", "confidence": 1.0})

                # Guarantee that recognized named individuals appearing in video are included in tags
                existing_people_tags = {
                    (t.get("tag") or "").strip().lower()
                    for t in serialized_tags if isinstance(t, dict)
                }
                for f_id, f_val in (face_intervals or {}).items():
                    if isinstance(f_val, dict):
                        name = (f_val.get("name") or "").strip()
                        if name and name.lower() not in ("unknown", "unnamed") and not name.lower().startswith(("face_", "face-", "person_")):
                            if name.lower() not in existing_people_tags:
                                serialized_tags.append({
                                    "tag": name,
                                    "category": "people",
                                    "confidence": 1.0
                                })
                                existing_people_tags.add(name.lower())
                        elif name and name != f_id and not f_id.startswith(("manual_", "face_manual_")):
                            if name.lower() not in existing_people_tags:
                                serialized_tags.append({
                                    "tag": name,
                                    "category": "people",
                                    "confidence": 1.0
                                })
                                existing_people_tags.add(name.lower())

            # 6. Duplicates calculation for video (pHash on first frame if requested)
            phash = existing_metadata.get("phash")
            if "duplicates" in active_modes or not phash:
                try:
                    first_frame = frames[0] if frames else (local_path if local_path.suffix.lower() in config.SUPPORTED_PHOTO_EXTS else None)
                    if first_frame:
                        phash = hash.calculate_phash(first_frame)
                except Exception:
                    pass

            # 7. Metadata assembly with soft merging
            metadata = dict(existing_metadata)
            metadata.update({
                "file_path": file_path,
                "file_name": filename,
                "file_size": file_size,
                "mtime": mtime,
                "media_type": "video",
                "content_type": gemini_dict.get("content_type") or metadata.get("content_type", "other"),
                "tags": serialized_tags,
                "video_metadata": video_meta,
                "face_timeline": face_intervals,
                "gemini_analysis": gemini_dict,
                "transcription": transcription,
                "transcription_ru": transcription_ru,
                "phash": phash,
                "modules_status": get_file_modules_status({
                    "transcription": transcription,
                    "face_timeline": face_intervals,
                    "phash": phash,
                    "gemini_analysis": gemini_dict
                }, "video")
            })

            # 8. Save sidecar file
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            # 9. Ingest into relational SQLite tables
            try:
                media_id = hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:16]
                database.upsert_media_item(
                    media_id=media_id,
                    file_path=file_path,
                    file_name=filename,
                    media_type="video",
                    file_size=file_size,
                    mtime=mtime,
                    duration=video_meta.get("duration") if video_meta else None,
                    media_date=video_meta.get("creation_time") if video_meta else None,
                    status="PROCESSED"
                )
                database.upsert_media_metadata(
                    media_id=media_id,
                    summary=gemini_dict.get("summary") or metadata.get("summary"),
                    summary_ru=gemini_dict.get("summary_ru") or metadata.get("summary_ru"),
                    description=gemini_dict.get("description") or metadata.get("description"),
                    description_ru=gemini_dict.get("description_ru") or metadata.get("description_ru"),
                    environment=gemini_dict.get("environment"),
                    lighting=gemini_dict.get("lighting"),
                    weather=gemini_dict.get("weather"),
                    time_of_day=gemini_dict.get("time_of_day"),
                    ocr_text=gemini_dict.get("ocr_text"),
                    location_name=gemini_dict.get("location_name"),
                    transcription=transcription,
                    transcription_ru=transcription_ru,
                    raw_exif=video_meta,
                    raw_gemini=gemini_dict if "error" not in gemini_dict else None
                )
                if face_intervals:
                    video_faces_list = []
                    for f_id, f_val in face_intervals.items():
                        video_faces_list.append({
                            "face_id": f_id,
                            "name": f_val.get("name", f_id),
                            "confidence": 1.0
                        })
                    database.save_media_faces(media_id, video_faces_list)
                if gemini_dict.get("timeline_events") and isinstance(gemini_dict["timeline_events"], list):
                    database.save_video_timeline_events(media_id, gemini_dict["timeline_events"])
                elif gemini_dict.get("timeline") and isinstance(gemini_dict["timeline"], list):
                    database.save_video_timeline_events(media_id, gemini_dict["timeline"])
                if serialized_tags:
                    database.save_media_tags(media_id, serialized_tags)
            except Exception as db_err:
                print(f"Warning: Failed to upsert relational video item for {filename}: {db_err}")

            # 10. Successful database record
            database.update_sync_record(
                file_path=file_path,
                file_size=file_size,
                mtime=mtime,
                status="PROCESSED",
                sidecar_path=str(sidecar_path)
            )
            print(f"Successfully processed: {filename} -> {sidecar_path.name}")
            return metadata

    except Exception as e:
        print(f"Error processing video {filename or file_path}: {e}")
        traceback.print_exc()
        database.update_sync_record(
            file_path=file_path,
            file_size=file_size,
            mtime=mtime,
            status="FAILED",
            error_message=str(e)
        )
        return None
    finally:
        if is_temp_download and local_path_str and os.path.isfile(local_path_str):
            try:
                os.unlink(local_path_str)
            except Exception:
                pass


# --- Duplicate Grouping Search ---

def process_duplicate_groups(
    processed_photos: List[Dict[str, Any]],
    ui_base_url: Optional[str] = None
):
    """Group similar photos and update their sidecar files with best frame selection results."""
    if len(processed_photos) < 2:
        return
        
    print("\n--- Starting Stage 3: Burst shooting and duplicate analysis ---")
    
    # 1. Prepare metadata for clustering
    files_meta = []
    for p in processed_photos:
        files_meta.append({
            "file_path": p["file_path"],
            "datetime": p["exif"]["datetime"],
            "phash": p["phash"]
        })
        
    # 2. Perform clustering
    file_to_group = hash.cluster_duplicates(files_meta)
    if not file_to_group:
        print("No burst duplicates or similar frames detected.")
        return
        
    groups: Dict[str, List[Path]] = {}
    for path_str, group_id in file_to_group.items():
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(Path(path_str))
        
    print(f"Found groups of similar frames: {len(groups)}")
    
    effective_ui_base = ui_base_url or getattr(config, "UI_BASE_URL", None)

    # 3. Analyze each group
    for group_id, image_paths in groups.items():
        print(f"\nAnalyzing group {group_id} ({len(image_paths)} photos)...")
        resolved_image_paths = []
        temp_files_to_clean = []
        try:
            for p in image_paths:
                try:
                    res_p = resolve_or_stream_media_file(str(p), ui_base_url=effective_ui_base)
                    resolved_image_paths.append(Path(res_p))
                    if os.path.abspath(res_p) != os.path.abspath(str(p)):
                        temp_files_to_clean.append(res_p)
                except Exception as load_err:
                    print(f"Warning: Could not resolve/stream image {p} for duplicate analysis in group {group_id}: {load_err}")

            if len(resolved_image_paths) < 2:
                print(f"Skipping duplicate group {group_id} (fewer than 2 accessible photos available).")
                continue

            # Evaluate via Gemini API
            gemini_res = retry_api_call(gemini.analyze_duplicates, resolved_image_paths)
            
            # 4. Update sidecar files for each image in group
            for path in image_paths:
                photo_meta = next((p for p in processed_photos if p["file_path"] == str(path)), None)
                if not photo_meta:
                    continue
                
                sync_rec = database.get_sync_record(str(path))
                if not sync_rec or not sync_rec["sidecar_path"]:
                    continue
                
                sidecar_path = Path(sync_rec["sidecar_path"])
                if not sidecar_path.is_file():
                    continue
                
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                is_best = (path.name == gemini_res.best_image_name)
                file_defects = next((d for d in gemini_res.defects if d.filename == path.name), None)
                
                data["duplicate_analysis"] = {
                    "group_id": group_id,
                    "changes_weight": gemini_res.changes_weight,
                    "is_best_in_group": is_best,
                    "best_image_name": gemini_res.best_image_name,
                    "reasoning": gemini_res.reasoning,
                    "reasoning_ru": gemini_res.reasoning_ru,
                    "defects": file_defects.model_dump() if file_defects else None
                }
                
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    
            print(f"Group {group_id} successfully processed. Best frame: {gemini_res.best_image_name}")
            
        except Exception as e:
            print(f"Error during duplicate group analysis {group_id}: {e}")
            traceback.print_exc()
        finally:
            for tf in temp_files_to_clean:
                try:
                    if os.path.isfile(tf):
                        os.unlink(tf)
                except Exception:
                    pass


# --- Worker Controls & Events ---

_pause_event = threading.Event()
_pause_event.set()
_stop_event = threading.Event()
_progress_callback = None

def set_progress_callback(callback):
    """Register a progress callback: fn(current, total, percent, current_file, stage)"""
    global _progress_callback
    _progress_callback = callback

def report_progress(current: int, total: int, current_file: str = "", stage: str = ""):
    """Report progress to registered callback."""
    global _progress_callback
    percent = int((current / total * 100)) if total > 0 else 0
    if _progress_callback:
        try:
            _progress_callback(current, total, percent, current_file, stage)
        except Exception:
            pass

def reset_controls():
    """Reset pause and stop events before starting a new task."""
    _stop_event.clear()
    _pause_event.set()

def request_pause():
    """Pause execution."""
    _pause_event.clear()

def request_resume():
    """Resume execution."""
    _pause_event.set()

def request_stop():
    """Request stop/cancellation of current task."""
    _stop_event.set()
    _pause_event.set()

def is_stop_requested() -> bool:
    """Check if stop was requested."""
    return _stop_event.is_set()

def is_paused() -> bool:
    """Check if worker is currently paused."""
    return not _pause_event.is_set()

def check_pause_and_stop() -> bool:
    """Checks stop event and waits if paused. Returns True if execution should continue."""
    while not _pause_event.is_set():
        if _stop_event.is_set():
            return False
        time.sleep(0.2)
    return not _stop_event.is_set()


# --- Media Worker Queue ---

class MediaWorkerQueue:
    """
    Concurrent Worker Queue for parallel media analysis.
    Supports dynamic concurrency and selective AI module modes.
    """
    def __init__(self, max_workers: Optional[int] = None, modes: Optional[Union[str, List[str], set]] = None):
        self.max_workers = max_workers or config.get_max_workers()
        self.modes = normalize_modes(modes)
        self.lock = threading.Lock()
        self.total_items = 0
        self.completed_items = 0
        self.failed_items = 0
        self.active_items: Dict[int, str] = {}
        self.processed_photos_meta: List[Dict[str, Any]] = []

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            active_files = list(self.active_items.values())
            pending = max(0, self.total_items - self.completed_items - len(active_files))
            return {
                "max_workers": self.max_workers,
                "active_workers": len(active_files),
                "in_flight_files": active_files,
                "pending_count": pending,
                "completed": self.completed_items,
                "failed": self.failed_items,
                "total": self.total_items,
                "modes": list(self.modes)
            }

    def process_queue(self, items: List[Any]) -> List[Dict[str, Any]]:
        """Processes list of items concurrently."""
        self.total_items = len(items)
        self.completed_items = 0
        self.failed_items = 0
        self.active_items.clear()
        self.processed_photos_meta.clear()

        if not items:
            report_progress(0, 0, "", "No files to process.")
            return []

        mode_names = []
        if "transcribe" in self.modes: mode_names.append("🎙️ Audio")
        if "faces" in self.modes: mode_names.append("👤 Faces")
        if "duplicates" in self.modes: mode_names.append("🗂️ Duplicates")
        if "vision" in self.modes: mode_names.append("🖼️ Vision")
        mode_str = " + ".join(mode_names) if len(mode_names) < 4 else "Full Pipeline"

        provider_name = config.MODEL_PROVIDER.capitalize()
        report_progress(
            0, self.total_items, "",
            f"Starting [{mode_str}] ({provider_name}) with {self.max_workers} worker(s)..."
        )
        print(f"\n[Worker Queue] Initialized [{mode_str}] with {self.max_workers} worker(s). Total items: {self.total_items}")

        def worker_task(item: Any, worker_id: int):
            if not check_pause_and_stop():
                return None

            file_path, folder, filename, file_size, mtime, stream_url, local_path, content_base64 = _extract_item_info(item)
            ext = Path(file_path).suffix.lower()

            if isinstance(item, tuple) and len(item) == 3:
                media_type = item[2]
            elif isinstance(item, dict) and "media_type" in item:
                media_type = item["media_type"]
            else:
                media_type = "photo" if ext in config.SUPPORTED_PHOTO_EXTS else "video"

            with self.lock:
                self.active_items[worker_id] = filename
                active_names = list(self.active_items.values())
                current_active_names = ", ".join(active_names[:3])
                if len(active_names) > 3:
                    current_active_names += f" (+{len(active_names) - 3} more)"
                report_progress(
                    self.completed_items,
                    self.total_items,
                    current_active_names,
                    f"[{mode_str}] Processing ({len(active_names)} active) [{self.completed_items}/{self.total_items}]"
                )

            res = None
            try:
                if isinstance(item, tuple) and len(item) >= 2:
                    call_item = item[0]
                    root_arg = item[1]
                else:
                    call_item = item
                    root_arg = Path(folder) if folder else None

                if media_type == "photo":
                    res = process_photo(call_item, root_arg, modes=self.modes)
                else:
                    res = process_video(call_item, root_arg, modes=self.modes)
            except Exception as e:
                print(f"[Worker {worker_id}] Error processing {filename}: {e}")

            finally:
                with self.lock:
                    self.active_items.pop(worker_id, None)
                    if res:
                        self.completed_items += 1
                        if media_type == "photo":
                            self.processed_photos_meta.append(res)
                    else:
                        self.failed_items += 1
                        self.completed_items += 1

                    active_names = list(self.active_items.values())
                    current_active_names = ", ".join(active_names[:3]) if active_names else ""
                    if len(active_names) > 3:
                        current_active_names += f" (+{len(active_names) - 3} more)"
                    report_progress(
                        self.completed_items,
                        self.total_items,
                        current_active_names,
                        f"[{mode_str}] Processed {self.completed_items}/{self.total_items} item(s)"
                    )

            return res

        worker_id_counter = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for item in items:
                if is_stop_requested():
                    break
                worker_id_counter += 1
                fut = executor.submit(worker_task, item, worker_id_counter)
                futures.append(fut)

            for fut in as_completed(futures):
                if is_stop_requested():
                    for f in futures:
                        f.cancel()
                    break
                try:
                    fut.result()
                except Exception as e:
                    print(f"Worker task error: {e}")

        return self.processed_photos_meta


_active_worker_queue: Optional[MediaWorkerQueue] = None
_active_queue_lock = threading.Lock()

def get_queue_status() -> Dict[str, Any]:
    """Retrieve status of active or last worker queue."""
    global _active_worker_queue
    with _active_queue_lock:
        if _active_worker_queue:
            return _active_worker_queue.get_status()
        return {
            "max_workers": config.get_max_workers(),
            "active_workers": 0,
            "in_flight_files": [],
            "pending_count": 0,
            "completed": 0,
            "failed": 0,
            "total": 0
        }

def get_models_loaded_status() -> Dict[str, bool]:
    """Check readiness of face detection/recognition, whisper, and vision LLM models."""
    face_detection = False
    face_recognition = False
    try:
        from src.utils import faces
        if getattr(faces, "_face_app", None) is not None:
            face_detection = True
            face_recognition = True
        elif getattr(faces, "_opencv_cascade", None) is not None:
            face_detection = True
            face_recognition = False
        else:
            try:
                import insightface
                face_detection = True
                face_recognition = True
            except ImportError:
                import cv2
                face_detection = True
                face_recognition = False
    except Exception:
        pass

    whisper_ready = False
    try:
        import faster_whisper
        whisper_ready = True
    except Exception:
        pass

    vision_llm_ready = False
    try:
        if config.MODEL_PROVIDER == "gemini":
            vision_llm_ready = bool(getattr(config, "GEMINI_API_KEY", None) or os.environ.get("GEMINI_API_KEY"))
        else:
            vision_llm_ready = bool(getattr(config, "LOCAL_API_BASE", None) or os.environ.get("LOCAL_API_BASE"))
    except Exception:
        pass

    return {
        "face_detection": face_detection,
        "face_recognition": face_recognition,
        "whisper": whisper_ready,
        "vision_llm": vision_llm_ready
    }


# --- Pipeline Orchestrator ---

def run_pipeline(
    force_reprocess: bool = False,
    max_workers: Optional[int] = None,
    files: Optional[List[Any]] = None,
    output_folder: Optional[str] = None,
    settings: Optional[Any] = None,
    modes: Optional[Union[str, List[str], set]] = None
):
    """
    Main method to launch media archive synchronization and cataloging via parallel worker queue.
    Supports modular modes (transcribe, faces, duplicates, vision) and smart skip.
    Supports stateless execution when files and dynamic settings are provided directly by UI.
    """
    global _active_worker_queue
    reset_controls()

    # Apply runtime settings if provided
    if settings:
        if isinstance(settings, dict):
            config.apply_runtime_settings(**settings)
            if modes is None and "modes" in settings:
                modes = settings.get("modes")
        elif hasattr(settings, "model_dump"):
            dump = settings.model_dump()
            config.apply_runtime_settings(**dump)
            if modes is None and "modes" in dump:
                modes = dump.get("modes")
        elif hasattr(settings, "__dict__"):
            config.apply_runtime_settings(**settings.__dict__)
            if modes is None and hasattr(settings, "modes"):
                modes = getattr(settings, "modes")

    if output_folder:
        config.apply_runtime_settings(output_folder=output_folder)

    active_modes = normalize_modes(modes)
    mode_display = ", ".join(active_modes)
    report_progress(0, 0, "", f"Initializing system and databases ({mode_display})...")
    database.init_db()
    if "faces" in active_modes:
        faces.init_face_analyzer()

    if not check_pause_and_stop():
        print("\n[STOPPED] Synchronization cancelled before scan.")
        return

    items_to_process: List[Any] = []
    skipped_count = 0

    if files is not None and len(files) > 0:
        # Stateless mode: Process files supplied directly by UI without scanning INPUT_FOLDERS
        print(f"Using provided file list ({len(files)} files) for modes [{mode_display}]...")
        for item in files:
            file_path, folder, filename, file_size, mtime, stream_url, local_path, content_base64 = _extract_item_info(item)
            ext = Path(file_path).suffix.lower()
            if ext not in config.SUPPORTED_PHOTO_EXTS and ext not in config.SUPPORTED_VIDEO_EXTS:
                continue

            media_type = "photo" if ext in config.SUPPORTED_PHOTO_EXTS else "video"
            need_processing = True

            if not force_reprocess:
                sync_rec = database.get_sync_record(file_path)
                if sync_rec and sync_rec["status"] == "PROCESSED" and sync_rec["sidecar_path"]:
                    sidecar = Path(sync_rec["sidecar_path"])
                    if sidecar.is_file():
                        size_match = (file_size == 0 or sync_rec["file_size"] == file_size)
                        mtime_match = (mtime == 0.0 or abs(sync_rec["mtime"] - mtime) < 0.01)
                        if size_match and mtime_match:
                            try:
                                with open(sidecar, "r", encoding="utf-8") as sc_f:
                                    sc_data = json.load(sc_f)
                                mod_status = sc_data.get("modules_status") or get_file_modules_status(sc_data, media_type)
                                all_satisfied = True
                                for m in active_modes:
                                    if m == "transcribe" and media_type == "photo":
                                        continue
                                    if not mod_status.get(m):
                                        all_satisfied = False
                                        break
                                if all_satisfied:
                                    need_processing = False
                                    skipped_count += 1
                            except Exception:
                                pass

            if need_processing:
                items_to_process.append({
                    "file_path": file_path,
                    "folder": folder,
                    "filename": filename,
                    "file_size": file_size,
                    "mtime": mtime,
                    "stream_url": stream_url,
                    "media_type": media_type
                })
    else:
        # Fallback to local scan for CLI/test workflows
        report_progress(0, 0, "", "Scanning media folders...")
        media_files = scan_input_folders()
        print(f"Found files to scan: {len(media_files)} for modes [{mode_display}]")
        for path, root in media_files:
            stat = path.stat()
            sync_rec = database.get_sync_record(str(path))
            need_processing = True
            ext = path.suffix.lower()
            media_type = "photo" if ext in config.SUPPORTED_PHOTO_EXTS else "video"

            if not force_reprocess and sync_rec:
                if sync_rec["status"] == "PROCESSED" and sync_rec["sidecar_path"]:
                    sidecar = Path(sync_rec["sidecar_path"])
                    if (sync_rec["file_size"] == stat.st_size and 
                        abs(sync_rec["mtime"] - stat.st_mtime) < 0.01 and 
                        sidecar.is_file()):
                        try:
                            with open(sidecar, "r", encoding="utf-8") as sc_f:
                                sc_data = json.load(sc_f)
                            mod_status = sc_data.get("modules_status") or get_file_modules_status(sc_data, media_type)
                            all_satisfied = True
                            for m in active_modes:
                                if m == "transcribe" and media_type == "photo":
                                    continue
                                if not mod_status.get(m):
                                    all_satisfied = False
                                    break
                            if all_satisfied:
                                need_processing = False
                                skipped_count += 1
                        except Exception:
                            pass

            if need_processing:
                if media_type == "photo":
                    items_to_process.append((path, root, "photo"))
                elif media_type == "video":
                    items_to_process.append((path, root, "video"))

    if skipped_count > 0:
        print(f"Skipped files with existing module results: {skipped_count}")

    total_items = len(items_to_process)
    if total_items == 0:
        report_progress(0, 0, "", f"All files up-to-date for [{mode_display}].")
        print(f"No new files to process for [{mode_display}].")
        return

    workers_count = max_workers or config.get_max_workers()
    queue = MediaWorkerQueue(max_workers=workers_count, modes=active_modes)
    with _active_queue_lock:
        _active_worker_queue = queue

    queue.process_queue(items_to_process)

    if not check_pause_and_stop():
        print("\n[STOPPED] Synchronization cancelled by user.")
        report_progress(total_items, total_items, "", "Stopped by user")
        return

    # Duplicate grouping on all processed photos if duplicates mode requested
    if "duplicates" in active_modes:
        report_progress(total_items, total_items, "", "Analyzing duplicate groups...")
        all_processed_photos = []
        sync_records = database.get_all_sync_records()
        for file_path, r in sync_records.items():
            if r.get("status") == "PROCESSED":
                sidecar_str = r.get("sidecar_path")
                if sidecar_str:
                    sidecar_path = Path(sidecar_str)
                    if sidecar_path.is_file():
                        try:
                            with open(sidecar_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                                if meta.get("media_type") == "photo":
                                    all_processed_photos.append(meta)
                        except Exception:
                            pass

        if len(all_processed_photos) >= 2:
            try:
                process_duplicate_groups(all_processed_photos)
            except Exception as dup_err:
                print(f"Warning: Duplicate analysis failed: {dup_err}")

    report_progress(total_items, total_items, "", f"Pipeline [{mode_display}] completed successfully")
    print(f"\nPipeline [{mode_display}] completed successfully.")


def find_media_file(target: str) -> Optional[Tuple[Path, Path]]:
    """
    Finds a media file matching the target string.
    Returns a tuple (file_path, input_root) or None if not found.
    """
    target_path = Path(target)
    
    if target_path.is_file():
        resolved_path = target_path.resolve()
        for folder in config.INPUT_FOLDERS:
            try:
                resolved_folder = folder.resolve()
                if resolved_path.is_relative_to(resolved_folder):
                    return resolved_path, folder
            except ValueError:
                pass
        return resolved_path, resolved_path.parent

    matches = []
    for folder in config.INPUT_FOLDERS:
        if not folder.exists():
            continue
        for root, dirs, files in os.walk(str(folder)):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if file.startswith("."):
                    continue
                path = Path(root) / file
                target_str = target.replace("\\", "/").lower()
                rel_to_folder = ""
                try:
                    rel_to_folder = str(path.relative_to(folder)).replace("\\", "/").lower()
                except ValueError:
                    pass
                
                if path.name.lower() == target.lower() or (rel_to_folder and rel_to_folder.endswith(target_str)):
                    matches.append((path, folder))
                    
    if not matches:
        return None
        
    if len(matches) > 1:
        print(f"Warning: Multiple matches found for '{target}':")
        for p, _ in matches:
            print(f"  - {p}")
        print(f"Selecting the first match: {matches[0][0]}")
        
    return matches[0]


def analyze_single_file(
    file_identifier: str,
    item_data: Optional[Any] = None,
    output_folder: Optional[str] = None,
    settings: Optional[Any] = None,
    modes: Optional[Union[str, List[str], set]] = None
) -> Optional[Dict[str, Any]]:
    """
    Main method to analyze a single media file, ignoring database status,
    and updating database records and duplicate groups.
    Supports modular modes (transcribe, faces, duplicates, vision).
    """
    if settings:
        if isinstance(settings, dict):
            config.apply_runtime_settings(**settings)
            if modes is None and "modes" in settings:
                modes = settings.get("modes")
        elif hasattr(settings, "model_dump"):
            dump = settings.model_dump()
            config.apply_runtime_settings(**dump)
            if modes is None and "modes" in dump:
                modes = dump.get("modes")
        elif hasattr(settings, "__dict__"):
            config.apply_runtime_settings(**settings.__dict__)
            if modes is None and hasattr(settings, "modes"):
                modes = getattr(settings, "modes")

    if output_folder:
        config.apply_runtime_settings(output_folder=output_folder)

    active_modes = normalize_modes(modes)

    database.init_db()
    if "faces" in active_modes:
        faces.init_face_analyzer()

    if item_data is not None:
        file_path, folder, filename, file_size, mtime, stream_url, local_path, content_base64 = _extract_item_info(item_data)
        if not file_path:
            file_path = file_identifier
    else:
        file_path = file_identifier
        folder = ""
        filename = Path(file_path.replace("\\", "/")).name
        file_size = 0
        mtime = 0.0
        stream_url = None
        local_path = None
        content_base64 = None

        match = find_media_file(file_identifier)
        if match:
            file_path = str(match[0])
            folder = str(match[1])
            filename = match[0].name

    ext = Path(file_path.replace("\\", "/")).suffix.lower()
    is_photo = ext in config.SUPPORTED_PHOTO_EXTS
    is_video = ext in config.SUPPORTED_VIDEO_EXTS

    if not is_photo and not is_video:
        print(f"Error: Unsupported file format '{ext}'.")
        return None

    if item_data is not None:
        call_item = {
            "file_path": file_path,
            "folder": folder,
            "filename": filename,
            "file_size": file_size,
            "mtime": mtime,
            "stream_url": stream_url,
            "local_path": local_path,
            "content_base64": content_base64
        }
        root_arg = Path(folder) if folder else None
    else:
        call_item = match[0] if match else Path(file_path)
        root_arg = match[1] if match else (Path(folder) if folder else None)

    target_tags_arg = None
    tag_format_arg = None
    custom_prompt_arg = None
    if settings:
        if isinstance(settings, dict):
            target_tags_arg = settings.get("target_tags")
            tag_format_arg = settings.get("tag_format")
            custom_prompt_arg = settings.get("custom_prompt") or settings.get("vision_prompt_template")
        elif hasattr(settings, "target_tags"):
            target_tags_arg = getattr(settings, "target_tags", None)
            tag_format_arg = getattr(settings, "tag_format", None)
            custom_prompt_arg = getattr(settings, "custom_prompt", None) or getattr(settings, "vision_prompt_template", None)

    call_kwargs = {
        "modes": active_modes
    }
    if target_tags_arg is not None:
        call_kwargs["target_tags"] = target_tags_arg
    if tag_format_arg is not None:
        call_kwargs["tag_format"] = tag_format_arg
    if custom_prompt_arg is not None:
        call_kwargs["custom_prompt"] = custom_prompt_arg

    meta = None
    if is_photo:
        photo_kwargs = {}
        try:
            sig = inspect.signature(process_photo)
            for k, v in call_kwargs.items():
                if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    photo_kwargs[k] = v
        except Exception:
            photo_kwargs = call_kwargs

        meta = process_photo(call_item, root_arg, **photo_kwargs)
    else:
        video_kwargs = {}
        try:
            sig = inspect.signature(process_video)
            for k, v in call_kwargs.items():
                if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    video_kwargs[k] = v
        except Exception:
            video_kwargs = call_kwargs

        meta = process_video(call_item, root_arg, **video_kwargs)

    if meta is None:
        print(f"\n[ERROR] Analysis for {filename or file_path} failed.")
        return None

    print(f"\nAnalysis for {filename or file_path} completed successfully.")
    return meta

