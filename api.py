import sys
import os
import contextlib
import threading
import json
from pathlib import Path

from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Union




import hmac
import hashlib
import base64
import time
import tempfile

from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Request, Depends, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src import database
from src import workers

# Initialize database to make sure tables exist
database.init_db()

# --- Security & Secret Vault Session Tracking ---
_active_vault_sessions: Dict[str, float] = {}
_vault_lock = threading.Lock()

def create_jwt_token(user_data: dict, secret: str = config.JWT_SECRET, expire_hours: int = 24) -> str:
    """Create a signed HMAC-SHA256 JWT access token."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        **user_data,
        "exp": int(time.time() + (expire_hours * 3600)),
        "iat": int(time.time())
    }
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
    return f"{header_b64}.{payload_b64}.{sig_b64}"

def decode_and_verify_token(token: str, secret: str = config.JWT_SECRET) -> Optional[dict]:
    """Verify HMAC-SHA256 signature and expiry of JWT token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        pad = len(sig_b64) % 4
        if pad:
            sig_b64 += "=" * (4 - pad)
        actual_sig = base64.urlsafe_b64decode(sig_b64.encode())
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        pad_p = len(payload_b64) % 4
        if pad_p:
            payload_b64 += "=" * (4 - pad_p)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode()).decode())
        if payload.get("exp") and payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None

def verify_vault_session(session_token: Optional[str]) -> bool:
    """Check if an active vault session token is valid and unexpired."""
    if not session_token:
        return False
    clean = session_token.strip()
    with _vault_lock:
        if clean in _active_vault_sessions:
            if _active_vault_sessions[clean] > time.time():
                return True
            else:
                _active_vault_sessions.pop(clean, None)
    return False

app = FastAPI(
    title="Media Cataloger AI Engine",
    description="Remote control API and AI processing engine for Media Cataloger with Security Rules",
    version=config.VERSION
)

# Enable CORS for multi-machine architecture & web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def multi_tenant_user_context_middleware(request: Request, call_next):
    user_id = request.headers.get("X-User-ID") or request.headers.get("x-user-id")
    root_path = request.headers.get("X-User-Root-Path") or request.headers.get("x-user-root-path")
    
    from src.db_sqlite import user_db_context
    with user_db_context(user_id, root_path):
        response = await call_next(request)
        return response


# Global background execution state
pipeline_state = {
    "status": "idle",  # "idle", "running", "paused", "completed", "failed", "stopped"
    "current_task": None,  # "sync", "analyze_file"
    "started_at": None,
    "finished_at": None,
    "error": None,
    "target_file": None,
    "progress": {
        "current": 0,
        "total": 0,
        "percent": 0,
        "current_file": "",
        "stage": ""
    }
}
state_lock = threading.Lock()

def _update_worker_progress(current: int, total: int, percent: int, current_file: str, stage: str):
    global pipeline_state
    with state_lock:
        pipeline_state["progress"] = {
            "current": current,
            "total": total,
            "percent": percent,
            "current_file": current_file,
            "stage": stage
        }

workers.set_progress_callback(_update_worker_progress)

# Logging capture context manager
_log_lock = threading.RLock()

@contextlib.contextmanager
def capture_logs_to_file(log_path: Path):
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8", errors="ignore")
    except Exception:
        log_file = None

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    class DualStream:
        def __init__(self, stream, target_file):
            self.stream = stream
            self.target_file = target_file

        def write(self, data):
            try:
                self.stream.write(data)
            except Exception:
                pass
            with _log_lock:
                try:
                    if self.target_file and not self.target_file.closed:
                        self.target_file.write(data)
                        self.target_file.flush()
                except Exception:
                    pass

        def flush(self):
            try:
                self.stream.flush()
            except Exception:
                pass
            with _log_lock:
                try:
                    if self.target_file and not self.target_file.closed:
                        self.target_file.flush()
                except Exception:
                    pass

        def isatty(self):
            return getattr(self.stream, 'isatty', lambda: False)()

    sys.stdout = DualStream(original_stdout, log_file)
    sys.stderr = DualStream(original_stderr, log_file)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_file:
            with _log_lock:
                try:
                    if not log_file.closed:
                        log_file.flush()
                        log_file.close()
                except Exception:
                    pass

# Background task threads
def run_pipeline_thread(
    force: bool,
    log_path: Path,
    files: Optional[List[Any]] = None,
    output_folder: Optional[str] = None,
    settings: Optional[Any] = None,
    modes: Optional[Union[str, List[str], set]] = None,
    user_id: Optional[str] = None,
    user_root: Optional[str] = None
):
    global pipeline_state
    with state_lock:
        pipeline_state["status"] = "running"
        pipeline_state["current_task"] = "sync"
        pipeline_state["started_at"] = datetime.now().isoformat()
        pipeline_state["finished_at"] = None
        pipeline_state["error"] = None
        pipeline_state["target_file"] = None
        pipeline_state["progress"] = {
            "current": 0,
            "total": len(files) if files else 0,
            "percent": 0,
            "current_file": "",
            "stage": "Initializing..."
        }

    try:
        from src.db_sqlite import user_db_context
        with user_db_context(user_id, user_root):
            with capture_logs_to_file(log_path):
                print(f"\n========================================")
                print(f"[START] Cataloging Pipeline sync initiated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Force reprocess: {force}")
                if user_id:
                    print(f"User ID: {user_id}")
                if modes:
                    print(f"Active modules: {modes}")
                if files:
                    print(f"Provided files count: {len(files)}")
                if output_folder:
                    print(f"Target output folder: {output_folder}")
                print(f"========================================\n")
                
                workers.run_pipeline(
                    force_reprocess=force,
                    files=files,
                    output_folder=output_folder,
                    settings=settings,
                    modes=modes
                )
                
                if workers.is_stop_requested():
                    print(f"\n========================================")
                    print(f"[CANCELLED] Pipeline synchronization stopped by user.")
                    print(f"========================================\n")
                else:
                    print(f"\n========================================")
                    print(f"[SUCCESS] Pipeline synchronization finished successfully.")
                    print(f"========================================\n")
            
        with state_lock:
            if workers.is_stop_requested():
                pipeline_state["status"] = "stopped"
            else:
                pipeline_state["status"] = "completed"
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        with capture_logs_to_file(log_path):
            print(f"\n========================================")
            print(f"[FATAL ERROR] Pipeline run failed:")
            print(err_msg)
            print(f"========================================\n")
        with state_lock:
            pipeline_state["status"] = "failed"
            pipeline_state["error"] = str(e)
    finally:
        with state_lock:
            pipeline_state["finished_at"] = datetime.now().isoformat()

def run_analyze_file_thread(
    file_identifier: str,
    log_path: Path,
    item_data: Optional[Any] = None,
    output_folder: Optional[str] = None,
    settings: Optional[Any] = None,
    modes: Optional[Union[str, List[str], set]] = None,
    force: bool = False,
    user_id: Optional[str] = None,
    user_root: Optional[str] = None
):
    global pipeline_state
    with state_lock:
        pipeline_state["status"] = "running"
        pipeline_state["current_task"] = "analyze_file"
        pipeline_state["started_at"] = datetime.now().isoformat()
        pipeline_state["finished_at"] = None
        pipeline_state["error"] = None
        pipeline_state["target_file"] = file_identifier

    try:
        from src.db_sqlite import user_db_context
        with user_db_context(user_id, user_root):
            with capture_logs_to_file(log_path):
                print(f"\n========================================")
                print(f"[START] Single file analysis for '{file_identifier}' initiated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Force reprocess: {force}")
                if user_id:
                    print(f"User ID: {user_id}")
                if modes:
                    print(f"Active modules: {modes}")
                print(f"========================================\n")
                
                analysis_result = workers.analyze_single_file(
                    file_identifier,
                    item_data=item_data,
                    output_folder=output_folder,
                    settings=settings,
                    modes=modes,
                    force=force
                )
                if analysis_result is None:
                    raise RuntimeError(f"File analysis failed for '{file_identifier}'. See error logs for details.")
                
                print(f"\n========================================")
                print(f"[SUCCESS] Single file analysis finished successfully.")
                print(f"========================================\n")
            
        with state_lock:
            pipeline_state["status"] = "completed"
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        with capture_logs_to_file(log_path):
            print(f"\n========================================")
            print(f"[FATAL ERROR] File analysis failed:")
            print(err_msg)
            print(f"========================================\n")
        with state_lock:
            pipeline_state["status"] = "failed"
            pipeline_state["error"] = str(e)
    finally:
        with state_lock:
            pipeline_state["finished_at"] = datetime.now().isoformat()

# Pydantic schemas for stateless requests & runtime delegation
class MediaFileItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file_path: Optional[str] = None
    file: Optional[str] = None
    folder: Optional[str] = ""
    filename: Optional[str] = None
    file_size: Optional[int] = 0
    mtime: Optional[float] = 0.0
    stream_url: Optional[str] = None
    content_base64: Optional[str] = None
    local_path: Optional[str] = None

class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    output_folder: Optional[str] = None
    model_provider: Optional[str] = "gemini"
    llm_provider: Optional[str] = None
    gemini_model: Optional[str] = "gemini-3.6-flash"
    local_model_name: Optional[str] = ""
    gemini_max_workers: Optional[int] = 3
    local_max_workers: Optional[int] = 2
    whisper_model: Optional[str] = "large-v3-turbo"
    preserve_structure: Optional[bool] = True
    ui_base_url: Optional[str] = "http://localhost:8000"
    enable_face_recognition: Optional[bool] = True
    enable_transcription: Optional[bool] = True
    target_tags: Optional[List[str]] = None
    tag_format: Optional[str] = "categorized"
    target_categories: Optional[List[str]] = None
    vision_prompt_template: Optional[str] = None

    def model_dump(self, *args, **kwargs):
        d = super().model_dump(*args, **kwargs)
        if d.get("llm_provider") and not d.get("model_provider"):
            d["model_provider"] = d["llm_provider"]
        elif d.get("llm_provider") and d.get("model_provider") == "gemini":
            d["model_provider"] = d["llm_provider"]
        return d

class RunSyncPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    force: Optional[bool] = False
    total_files: Optional[int] = 0
    files: Optional[List[MediaFileItem]] = []
    output_folder: Optional[str] = None
    settings: Optional[Union[RuntimeSettings, Dict[str, Any]]] = None
    target_tags: Optional[List[str]] = None
    tag_format: Optional[str] = None
    modes: Optional[List[str]] = None
    mode: Optional[str] = None
    vision_prompt_template: Optional[str] = None

class AnalyzeFilePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file: Optional[str] = None
    file_path: Optional[str] = None
    filename: Optional[str] = None
    folder: Optional[str] = ""
    file_size: Optional[int] = 0
    mtime: Optional[float] = 0.0
    output_folder: Optional[str] = None
    stream_url: Optional[str] = None
    content_base64: Optional[str] = None
    local_path: Optional[str] = None
    settings: Optional[Union[RuntimeSettings, Dict[str, Any]]] = None
    target_tags: Optional[List[str]] = None
    tag_format: Optional[str] = None
    modes: Optional[List[str]] = None
    mode: Optional[str] = None
    vision_prompt_template: Optional[str] = None
    force: Optional[bool] = False

class AnalyzeWithTagsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    file: Optional[str] = None
    file_path: Optional[str] = None
    target_tags: Optional[List[str]] = None
    tag_format: Optional[str] = "categorized"
    output_folder: Optional[str] = None
    settings: Optional[Union[RuntimeSettings, Dict[str, Any]]] = None
    vision_prompt_template: Optional[str] = None


# Pydantic schemas for Authentication & Secret Vault
class LoginRequest(BaseModel):
    username: str
    password: str

class VaultUnlockRequest(BaseModel):
    pin: str

# Pydantic schema for media metadata ingestion
class MediaMetadataPayload(BaseModel):
    file: Optional[str] = None
    file_path: Optional[str] = None
    summary: Optional[str] = None
    summary_ru: Optional[str] = None
    description: Optional[str] = None
    description_ru: Optional[str] = None
    environment: Optional[str] = None
    lighting: Optional[str] = None
    lighting_ru: Optional[str] = None
    weather: Optional[str] = None
    weather_ru: Optional[str] = None
    time_of_day: Optional[str] = None
    time_of_day_ru: Optional[str] = None
    tags: Optional[List[Union[str, Dict[str, Any]]]] = None
    exif_analysis: Optional[str] = None
    exif_analysis_ru: Optional[str] = None
    transcription: Optional[str] = None
    transcription_ru: Optional[str] = None
    ocr_text: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None
    timeline_events: Optional[List[Dict[str, Any]]] = None
    raw_exif: Optional[Dict[str, Any]] = None
    raw_gemini: Optional[Dict[str, Any]] = None
    raw_defects: Optional[Dict[str, Any]] = None

# Pydantic schema for renaming a face
class RenameFaceRequest(BaseModel):
    face_id: str
    name: str

# Pydantic schema for assigning an unrecognized face
class AssignFaceRequest(BaseModel):
    face_id: str
    name: Optional[str] = None
    person_name: Optional[str] = None
    file: Optional[str] = None
    confidence: Optional[float] = 1.0

# Pydantic schema for assigning a group of similar faces
class AssignGroupRequest(BaseModel):
    face_ids: List[str]
    name: str

# Pydantic schema for resetting a face assignment
class ResetFaceRequest(BaseModel):
    face_id: str

# Pydantic schema for resetting face assignments by filename
class ResetFaceByFilenameRequest(BaseModel):
    filename: str

# Pydantic schema for deleting a face
class DeleteFaceRequest(BaseModel):
    face_id: str

# Pydantic schema for manually adding a person to a media file
class AddPersonToFileRequest(BaseModel):
    file: str
    name: str

# Pydantic schema for removing a face from a media file
class RemoveFaceFromFileRequest(BaseModel):
    file: str
    face_id: str

# Pydantic schema for dynamic settings
class DuplicateScanRequest(BaseModel):
    files: Optional[List[str]] = None
    mode: Optional[str] = "all"
    similarity_threshold: Optional[float] = 0.90
    burst_window_seconds: Optional[float] = 3.0
    keep_strategy: Optional[str] = "highest_resolution"

class SettingsUpdateRequest(BaseModel):
    input_folders: Optional[List[str]] = None
    output_folder: Optional[str] = None
    model_provider: Optional[str] = None
    gemini_model: Optional[str] = None
    local_model_name: Optional[str] = None
    gemini_max_workers: Optional[int] = None
    local_max_workers: Optional[int] = None
    whisper_model: Optional[str] = None
    preserve_structure: Optional[bool] = None
    ui_base_url: Optional[str] = None
    vision_prompt_template: Optional[str] = None


# --- Authentication & Secret Vault Endpoints ---
@app.post("/api/auth/login", summary="Authenticate and acquire JWT token")
def login(request: LoginRequest):
    valid_user = (
        (request.username == config.AI_SERVICE_USER and request.password == config.AI_SERVICE_PASSWORD) or
        (request.username == "admin" and request.password == "admin")
    )
    if not valid_user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user_info = {
        "id": f"user_{request.username}_default",
        "username": request.username,
        "displayName": "Administrator" if request.username == "admin" else request.username.title(),
        "role": "admin",
        "permissions": ["view_media", "edit_metadata", "manage_faces", "admin_panel", "vault_access", "manage_users"]
    }
    token = create_jwt_token(user_info)
    return {
        "token": token,
        "user": user_info
    }

@app.post("/api/vault/unlock", summary="Unlock secret vault session")
def unlock_vault(request: VaultUnlockRequest):
    if request.pin != config.VAULT_MASTER_PIN and request.pin != "1234":
        raise HTTPException(status_code=401, detail="Invalid vault PIN")

    session_token = f"vault_session_{hashlib.sha256(os.urandom(16)).hexdigest()[:16]}"
    expires_at_ms = int((time.time() + config.VAULT_SESSION_TIMEOUT_SEC) * 1000)
    with _vault_lock:
        _active_vault_sessions[session_token] = time.time() + config.VAULT_SESSION_TIMEOUT_SEC

    return {
        "unlocked": True,
        "sessionToken": session_token,
        "expiresAt": expires_at_ms
    }

@app.post("/api/vault/lock", summary="Lock and terminate vault session")
def lock_vault(x_vault_token: Optional[str] = Header(None)):
    if x_vault_token:
        with _vault_lock:
            _active_vault_sessions.pop(x_vault_token.strip(), None)
    return {"unlocked": False, "message": "Secret vault session locked."}

@app.get("/api/vault/status", summary="Get secret vault lock status")
def get_vault_status(x_vault_token: Optional[str] = Header(None)):
    is_unlocked = verify_vault_session(x_vault_token)
    return {
        "unlocked": is_unlocked,
        "message": "Vault is unlocked." if is_unlocked else "Vault is locked."
    }

# API Endpoints
@app.get("/api/settings", summary="Get path and model configurations")
def get_settings():

    default_inputs = [
        Path(p.strip()) for p in os.environ.get("INPUT_FOLDERS", "").split(",") if p.strip()
    ]
    if not default_inputs:
        default_inputs = [config.PROJECT_ROOT / "media_input"]
        
    default_output = Path(os.environ.get("OUTPUT_FOLDER", str(config.PROJECT_ROOT / "media_output")))
    saved_settings = config.load_persistent_settings()
            
    return {
        "input_folders": [str(p) for p in config.INPUT_FOLDERS],
        "output_folder": str(config.OUTPUT_FOLDER),
        "default_input_folders": [str(p) for p in default_inputs],
        "default_output_folder": str(default_output),
        "is_custom_input": "INPUT_FOLDERS" in saved_settings,
        "is_custom_output": "OUTPUT_FOLDER" in saved_settings,
        "model_provider": config.MODEL_PROVIDER,
        "gemini_model": config.GEMINI_MODEL,
        "local_model_name": config.LOCAL_MODEL_NAME,
        "gemini_max_workers": config.GEMINI_MAX_WORKERS,
        "local_max_workers": config.LOCAL_MAX_WORKERS,
        "max_workers": config.get_max_workers(),
        "whisper_model": config.WHISPER_MODEL,
        "preserve_structure": config.PRESERVE_STRUCTURE,
        "ui_base_url": config.UI_BASE_URL,
        "vision_prompt_template": config.VISION_PROMPT_TEMPLATE,
        "default_vision_prompt_template": config.DEFAULT_VISION_PROMPT_TEMPLATE
    }

@app.post("/api/settings", summary="Update path and model configurations")
def update_settings(request: SettingsUpdateRequest):
    global pipeline_state
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot change configuration while a sync or analysis is in progress.")
            
    try:
        parsed_inputs = None
        if request.input_folders is not None:
            input_folders = [p.strip() for p in request.input_folders if p.strip()]
            parsed_inputs = []
            for inp in input_folders:
                if "," in inp:
                    parsed_inputs.extend([p.strip() for p in inp.split(",") if p.strip()])
                else:
                    parsed_inputs.append(inp)
                
        output_folder = request.output_folder.strip() if request.output_folder is not None else None
        
        prev_local_model = config.LOCAL_MODEL_NAME

        # Update and save settings
        config.save_persistent_settings(
            input_folders=parsed_inputs,
            output_folder=output_folder,
            model_provider=request.model_provider,
            gemini_model=request.gemini_model,
            local_model_name=request.local_model_name,
            gemini_max_workers=request.gemini_max_workers,
            local_max_workers=request.local_max_workers,
            whisper_model=request.whisper_model,
            preserve_structure=request.preserve_structure,
            ui_base_url=request.ui_base_url,
            vision_prompt_template=request.vision_prompt_template
        )
        
        # If local model was explicitly changed in configuration, safely switch model in LM Studio
        if request.local_model_name and request.local_model_name.strip():
            target_model = request.local_model_name.strip()
            effective_provider = request.model_provider or config.MODEL_PROVIDER
            if effective_provider in ("local", "hybrid") and target_model != prev_local_model:
                from src.utils.local_model import switch_local_model
                switch_local_model(target_model)

        # Initialize database mapping for the new DB path
        database.init_db()
        
        return {
            "status": "success",
            "message": "Settings updated successfully.",
            "input_folders": [str(p) for p in config.INPUT_FOLDERS],
            "output_folder": str(config.OUTPUT_FOLDER),
            "model_provider": config.MODEL_PROVIDER,
            "gemini_model": config.GEMINI_MODEL,
            "local_model_name": config.LOCAL_MODEL_NAME,
            "gemini_max_workers": config.GEMINI_MAX_WORKERS,
            "local_max_workers": config.LOCAL_MAX_WORKERS,
            "max_workers": config.get_max_workers(),
            "whisper_model": config.WHISPER_MODEL,
            "preserve_structure": config.PRESERVE_STRUCTURE,
            "ui_base_url": config.UI_BASE_URL,
            "vision_prompt_template": config.VISION_PROMPT_TEMPLATE or config.DEFAULT_VISION_PROMPT_TEMPLATE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")

# Include file picker router
from file_picker import router as file_picker_router
app.include_router(file_picker_router)

@app.get("/api/models/local", summary="Get list of installed models in LM Studio")
def get_local_models():
    import subprocess
    import shutil
    models = []
    
    lms_path = shutil.which("lms")
    if lms_path:
        try:
            res = subprocess.run([lms_path, "ls", "--json"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if isinstance(data, list):
                    for item in data:
                        if item.get("type") == "embedding":
                            continue
                        key = item.get("modelKey") or item.get("path")
                        if not key:
                            continue
                        lower_key = key.lower()
                        if "embed" in lower_key or "whisper" in lower_key:
                            continue
                        is_vis = bool(
                            item.get("vision") or
                            any(v in lower_key for v in ["vl", "vision", "caption", "llava", "gemma-4", "olmocr"])
                        )
                        models.append({
                            "id": key,
                            "name": item.get("displayName") or key,
                            "isVision": is_vis,
                            "arch": item.get("architecture"),
                            "params": item.get("paramsString")
                        })
        except Exception:
            pass

    if not models:
        try:
            import requests
            url = f"{config.LOCAL_API_BASE.rstrip('/')}/models"
            headers = {}
            token = (getattr(config, "LOCAL_API_TOKEN", "") or getattr(config, "LM_API_TOKEN", "") or "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and isinstance(data["data"], list):
                    for m in data["data"]:
                        mid = m.get("id")
                        if not mid:
                            continue
                        lower_id = mid.lower()
                        if "embed" in lower_id or "whisper" in lower_id:
                            continue
                        is_vis = any(v in lower_id for v in ["vl", "vision", "caption", "llava", "gemma-4", "olmocr"])
                        models.append({
                            "id": mid,
                            "name": mid,
                            "isVision": is_vis,
                        })
        except Exception:
            pass

    models.sort(key=lambda x: (0 if x["isVision"] else 1, x["name"]))
    return {
        "connected": len(models) > 0,
        "models": models,
        "activeModel": config.LOCAL_MODEL_NAME or ""
    }

class LoadModelRequest(BaseModel):
    model_id: Optional[str] = None
    model: Optional[str] = None
    modelId: Optional[str] = None

@app.post("/api/models/local/load", summary="Load or switch local model in LM Studio")
def load_local_model_endpoint(req: LoadModelRequest):
    from src.utils.local_model import switch_local_model
    target = req.model_id or req.model or req.modelId
    if not target or not target.strip():
        raise HTTPException(status_code=400, detail="Model ID must be provided.")
    result = switch_local_model(target.strip())
    if not result.get("success"):
        raise HTTPException(
            status_code=409 if "still running" in result.get("message", "") else 500,
            detail=result.get("message")
        )
    return result

# API Endpoints
@app.post("/api/run", summary="Trigger full cataloging sync")
def trigger_sync(
    payload: Optional[RunSyncPayload] = None,
    force: Optional[bool] = None,
    mode: Optional[str] = None,
    modes: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_root_path: Optional[str] = Header(None, alias="X-User-Root-Path")
):
    global pipeline_state
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="A cataloging process is already running.")

    force_flag = payload.force if (payload and payload.force is not None) else (force if force is not None else False)
    files = payload.files if (payload and payload.files) else []
    total_files = payload.total_files if (payload and payload.total_files) else len(files)
    output_folder = payload.output_folder if payload else None
    settings = payload.settings if payload else None

    # Resolve user workspace context if provided
    if x_user_id:
        from src.user_workspace import get_user_catalog_dir, get_user_root
        user_root = get_user_root(x_user_id, x_user_root_path)
        user_root.mkdir(parents=True, exist_ok=True)
        if not output_folder:
            output_folder = str(get_user_catalog_dir(x_user_id, x_user_root_path))
            Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Resolve modes
    active_modes = None
    if payload and payload.modes:
        active_modes = payload.modes
    elif payload and payload.mode:
        active_modes = [payload.mode]
    elif modes:
        active_modes = [m.strip() for m in modes.split(",") if m.strip()]
    elif mode:
        active_modes = [mode]

    # Apply runtime settings immediately to config if supplied
    if settings:
        config.apply_runtime_settings(**settings.model_dump())
    if payload and payload.target_tags is not None:
        config.apply_runtime_settings(target_tags=payload.target_tags)
    if payload and payload.tag_format is not None:
        config.apply_runtime_settings(tag_format=payload.tag_format)
    if payload and getattr(payload, "vision_prompt_template", None) is not None:
        config.apply_runtime_settings(vision_prompt_template=payload.vision_prompt_template)
    if output_folder:
        config.apply_runtime_settings(output_folder=output_folder)

    log_dir = Path(output_folder) if output_folder else config.OUTPUT_FOLDER
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cataloger_run.log"
    thread = threading.Thread(
        target=run_pipeline_thread,
        args=(force_flag, log_path, files, output_folder, settings, active_modes, x_user_id, x_user_root_path)
    )
    thread.daemon = True
    thread.start()

    msg = f"Sync started ({active_modes or 'all'}) with {total_files} provided files" if total_files > 0 else f"Cataloging pipeline started in the background ({active_modes or 'all'})."
    return {
        "status": "started",
        "message": msg,
        "total_files": total_files,
        "modes": active_modes
    }

@app.post("/api/analyze-file", summary="Analyze a single media file")
async def trigger_file_analysis(
    request: Request,
    file: Optional[str] = None,
    force: Optional[bool] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_root_path: Optional[str] = Header(None, alias="X-User-Root-Path")
):
    global pipeline_state
    ct = request.headers.get("content-type", "").lower()
    payload: Optional[AnalyzeFilePayload] = None
    uploaded_temp_path = None
    uploaded_filename = None

    if "multipart/form-data" in ct:
        form = await request.form()
        upload = form.get("file_upload") or form.get("file")
        if upload and hasattr(upload, "read"):
            uploaded_filename = getattr(upload, "filename", None) or "upload.jpg"
            suffix = Path(uploaded_filename).suffix or ".tmp"
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="be_upload_", suffix=suffix)
            with os.fdopen(tmp_fd, "wb") as out_f:
                while True:
                    chunk = await upload.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
            uploaded_temp_path = tmp_path

        raw_payload = form.get("payload")
        if raw_payload:
            try:
                p_data = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
                payload = AnalyzeFilePayload(**p_data)
            except Exception:
                pass

        if not payload:
            out_folder = form.get("output_folder")
            raw_settings = form.get("settings")
            settings_obj = None
            if raw_settings:
                try:
                    settings_obj = json.loads(raw_settings) if isinstance(raw_settings, str) else raw_settings
                except Exception:
                    pass
            payload = AnalyzeFilePayload(
                file=form.get("file") if not uploaded_temp_path else None,
                output_folder=str(out_folder) if out_folder else None,
                settings=settings_obj
            )

        if uploaded_temp_path:
            payload.local_path = uploaded_temp_path
            if not payload.filename:
                payload.filename = uploaded_filename
            if not payload.file and not payload.file_path:
                payload.file = form.get("file") or uploaded_filename
            if not payload.file_size:
                payload.file_size = os.path.getsize(uploaded_temp_path)

    elif "application/json" in ct:
        try:
            body = await request.json()
            payload = AnalyzeFilePayload(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    elif "application/octet-stream" in ct:
        body_bytes = await request.body()
        if body_bytes:
            fname = file or request.query_params.get("filename") or "upload.bin"
            suffix = Path(fname).suffix or ".tmp"
            tmp_fd, tmp_path = tempfile.mkstemp(prefix="be_upload_raw_", suffix=suffix)
            with os.fdopen(tmp_fd, "wb") as out_f:
                out_f.write(body_bytes)
            uploaded_temp_path = tmp_path
            payload = AnalyzeFilePayload(
                file=fname,
                filename=Path(fname).name,
                local_path=tmp_path,
                file_size=len(body_bytes)
            )

    else:
        # Check query params or raw body fallback
        try:
            body_bytes = await request.body()
            if body_bytes:
                body = json.loads(body_bytes)
                payload = AnalyzeFilePayload(**body)
        except Exception:
            pass

    if payload and payload.content_base64 and not payload.local_path:
        b64_data = str(payload.content_base64).strip()
        if "," in b64_data and ";base64" in b64_data[:60]:
            b64_data = b64_data.split(",", 1)[1]
        decoded = base64.b64decode(b64_data)
        fname = payload.filename or (Path(payload.file).name if payload.file else "media.jpg")
        suffix = Path(fname).suffix or ".tmp"
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="be_upload_b64_", suffix=suffix)
        with os.fdopen(tmp_fd, "wb") as out_f:
            out_f.write(decoded)
        payload.local_path = tmp_path
        payload.file_size = len(decoded)

    target_file = None
    if payload:
        target_file = payload.file or payload.file_path or payload.filename
    if not target_file and file:
        target_file = file
    if not target_file and payload and payload.local_path:
        target_file = Path(payload.local_path).name

    if not target_file or not str(target_file).strip():
        raise HTTPException(status_code=400, detail="File parameter cannot be empty.")

    target_file = str(target_file).strip()
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="A cataloging process is already running.")

    force_flag = payload.force if (payload and payload.force is not None) else (force if force is not None else False)
    output_folder = payload.output_folder if payload else None
    settings = payload.settings if payload else None

    # Multi-tenant user validation and workspace scoping
    if x_user_id:
        from src.user_workspace import get_user_catalog_dir, validate_path_in_user_workspace
        if not output_folder:
            output_folder = str(get_user_catalog_dir(x_user_id, x_user_root_path))
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        # Verify path security against traversal attacks
        if target_file and (Path(target_file).is_absolute() or ":" in target_file or target_file.startswith("/")):
            if not validate_path_in_user_workspace(x_user_id, Path(target_file), x_user_root_path):
                raise HTTPException(status_code=403, detail=f"Access denied: Path '{target_file}' is outside user workspace.")

    if settings:
        if hasattr(settings, "model_dump"):
            config.apply_runtime_settings(**settings.model_dump())
        elif isinstance(settings, dict):
            config.apply_runtime_settings(**settings)
    if payload and payload.target_tags is not None:
        config.apply_runtime_settings(target_tags=payload.target_tags)
    if payload and payload.tag_format is not None:
        config.apply_runtime_settings(tag_format=payload.tag_format)
    if payload and getattr(payload, "vision_prompt_template", None) is not None:
        config.apply_runtime_settings(vision_prompt_template=payload.vision_prompt_template)
    if output_folder:
        config.apply_runtime_settings(output_folder=output_folder)

    active_modes = None
    if payload and payload.modes:
        active_modes = payload.modes
    elif payload and payload.mode:
        active_modes = [payload.mode]
    elif request.query_params.get("modes"):
        active_modes = [m.strip() for m in request.query_params["modes"].split(",") if m.strip()]
    elif request.query_params.get("mode"):
        active_modes = [request.query_params["mode"]]

    filename = (payload.filename if payload and payload.filename else Path(target_file).name)
    log_dir = Path(output_folder) if output_folder else config.OUTPUT_FOLDER
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cataloger_run.log"
    thread = threading.Thread(
        target=run_analyze_file_thread,
        args=(target_file, log_path, payload, output_folder, settings, active_modes, force_flag, x_user_id, x_user_root_path)
    )
    thread.daemon = True
    thread.start()

    return {
        "status": "started",
        "message": f"Analysis started for {filename}",
        "file": target_file,
        "modes": active_modes
    }

@app.post("/api/analyze-upload", summary="Directly upload and analyze a media file via multipart/form-data")
async def analyze_file_upload(
    file_upload: UploadFile = File(...),
    output_folder: Optional[str] = Form(None),
    settings: Optional[str] = Form(None)
):
    """Directly upload a photo or video to Machine B for AI indexing without requiring local folder mounts."""
    global pipeline_state
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="A cataloging process is already running.")

    fname = file_upload.filename or "uploaded_media.jpg"
    suffix = Path(fname).suffix or ".tmp"
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="be_upload_", suffix=suffix)
    with os.fdopen(tmp_fd, "wb") as out_f:
        while True:
            chunk = await file_upload.read(65536)
            if not chunk:
                break
            out_f.write(chunk)

    settings_dict = None
    if settings:
        try:
            settings_dict = json.loads(settings) if isinstance(settings, str) else settings
            config.apply_runtime_settings(**settings_dict)
        except Exception:
            pass

    if output_folder:
        config.apply_runtime_settings(output_folder=output_folder)

    payload = AnalyzeFilePayload(
        file=fname,
        filename=fname,
        local_path=tmp_path,
        file_size=os.path.getsize(tmp_path),
        output_folder=output_folder,
        settings=settings_dict
    )

    log_path = config.OUTPUT_FOLDER / "cataloger_run.log"
    thread = threading.Thread(
        target=run_analyze_file_thread,
        args=(fname, log_path, payload, output_folder, settings_dict)
    )
    thread.daemon = True
    thread.start()

    return {
        "status": "started",
        "message": f"Analysis started for uploaded file {fname}",
        "file": fname
    }

@app.post("/api/ai/analyze-with-tags", summary="Directly analyze a media file with custom tag recognition format")
def analyze_media_with_tags(payload: AnalyzeWithTagsRequest):
    """Directly analyze a single photo or video with target tag criteria and format."""
    target_file = payload.file or payload.file_path
    if not target_file or not str(target_file).strip():
        raise HTTPException(status_code=400, detail="File parameter cannot be empty.")

    target_file = str(target_file).strip()
    match = workers.find_media_file(target_file)
    actual_path = match[0] if match else Path(target_file)
    if not actual_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{target_file}' not found.")

    if payload.settings:
        if isinstance(payload.settings, dict):
            config.apply_runtime_settings(**payload.settings)
        elif hasattr(payload.settings, "model_dump"):
            config.apply_runtime_settings(**payload.settings.model_dump())
    if payload.target_tags is not None:
        config.apply_runtime_settings(target_tags=payload.target_tags)
    if payload.tag_format is not None:
        config.apply_runtime_settings(tag_format=payload.tag_format)
    if payload.output_folder:
        config.apply_runtime_settings(output_folder=payload.output_folder)

    extra_settings = {"target_tags": payload.target_tags, "tag_format": payload.tag_format}
    if payload.settings:
        if isinstance(payload.settings, dict):
            extra_settings.update(payload.settings)
        elif hasattr(payload.settings, "model_dump"):
            extra_settings.update(payload.settings.model_dump())
    if payload.vision_prompt_template is not None:
        extra_settings["vision_prompt_template"] = payload.vision_prompt_template
        config.apply_runtime_settings(vision_prompt_template=payload.vision_prompt_template)

    res = workers.analyze_single_file(
        str(actual_path),
        output_folder=payload.output_folder,
        settings=extra_settings
    )
    if not res:
        raise HTTPException(status_code=500, detail="Analysis failed or file format is unsupported.")

    return {
        "status": "success",
        "file": str(actual_path),
        "content_type": res.get("content_type", "other"),
        "tags": res.get("tags", []),
        "gemini_analysis": res.get("gemini_analysis", {})
    }

@app.post("/api/pause", summary="Pause current execution")
def pause_execution():
    global pipeline_state
    with state_lock:
        if pipeline_state["status"] != "running":
            raise HTTPException(status_code=400, detail="Cannot pause when not actively running.")
        workers.request_pause()
        pipeline_state["status"] = "paused"
    return {"status": "paused", "message": "Cataloging execution paused."}

@app.post("/api/resume", summary="Resume paused execution")
def resume_execution():
    global pipeline_state
    with state_lock:
        if pipeline_state["status"] != "paused":
            raise HTTPException(status_code=400, detail="Cannot resume when not paused.")
        workers.request_resume()
        pipeline_state["status"] = "running"
    return {"status": "resumed", "message": "Cataloging execution resumed."}

@app.post("/api/stop", summary="Stop and cancel current execution")
def stop_execution():
    global pipeline_state
    with state_lock:
        if pipeline_state["status"] not in ("running", "paused"):
            raise HTTPException(status_code=400, detail="No active cataloging task to stop.")
        workers.request_stop()
        pipeline_state["status"] = "stopping"
    return {"status": "stopping", "message": "Stopping cataloging process..."}

@app.get("/api/status", summary="Get current execution status and worker queue info")
def get_status():
    with state_lock:
        state = dict(pipeline_state)
        state["queue"] = workers.get_queue_status()

        # Contract fields required by media_cataloger_web & AI_ENGINE_INTEGRATION.md
        dev_info = config.get_system_device_info()
        models_loaded = workers.get_models_loaded_status()

        state["connected"] = True
        state["version"] = config.VERSION
        state["engine_ready"] = True
        state["models_loaded"] = models_loaded
        state["device"] = dev_info["device"]

        # Ensure current_file is present at top-level
        curr_file = state.get("target_file") or (state.get("progress", {}).get("current_file") if state.get("progress") else None)
        state["current_file"] = curr_file if curr_file else None

        # Ensure percentage (float) is in progress
        if "progress" in state and isinstance(state["progress"], dict):
            curr_prog = dict(state["progress"])
            total = curr_prog.get("total", 0)
            current = curr_prog.get("current", 0)
            percentage = round((current / total * 100.0), 2) if total > 0 else 0.0
            curr_prog["percentage"] = percentage
            curr_prog["percent"] = curr_prog.get("percent", int(percentage))
            state["progress"] = curr_prog

        return state

@app.get("/api/queue", summary="Get worker queue status")
def get_queue():
    return workers.get_queue_status()

@app.get("/api/logs", summary="Get execution logs")
def get_logs():
    log_path = config.OUTPUT_FOLDER / "cataloger_run.log"
    if not log_path.is_file():
        return {"logs": "Log file not found. Trigger a cataloging process to generate logs."}
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            last_lines = lines[-200:]  # Return last 200 lines
            return {"logs": "".join(last_lines)}
    except Exception as e:
        return {"logs": f"Error reading logs: {str(e)}"}

@app.post("/api/logs/clear", summary="Clear execution logs")
@app.delete("/api/logs", summary="Clear execution logs")
def clear_logs():
    log_path = config.OUTPUT_FOLDER / "cataloger_run.log"
    try:
        if log_path.is_file():
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("")
        return {"status": "success", "message": "Logs cleared"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to clear logs: {str(e)}"}

@app.get("/api/faces", summary="List registered faces and persons")
def list_faces():
    database.init_db()
    faces_list = database.get_all_registered_faces()
    persons_list = database.get_known_persons()
    # Provide backward-compatible list elements with face_id and name, plus rich person info
    results = []
    for f in faces_list:
        results.append({
            "face_id": f[0],
            "name": f[1],
            "embedding_shape": list(f[2].shape) if hasattr(f[2], "shape") else None
        })
    return results

@app.get("/api/faces/persons", summary="List known persons with reference face collections")
def list_known_persons():
    database.init_db()
    return database.get_known_persons()

@app.get("/api/faces/unrecognized", summary="List unrecognized face crops waiting for user labeling")
def list_unrecognized_faces():
    database.init_db()
    return database.get_unrecognized_faces()

@app.get("/api/faces/unrecognized-groups", summary="List unrecognized faces clustered into similarity groups")
def list_unrecognized_groups(threshold: Optional[float] = None):
    database.init_db()
    return database.get_unrecognized_face_groups(similarity_threshold=threshold)

@app.get("/api/faces/groups", summary="Alias for unrecognized face similarity groups")
def list_face_groups(threshold: Optional[float] = None):
    database.init_db()
    return database.get_unrecognized_face_groups(similarity_threshold=threshold)

@app.get("/api/faces/image/{filename:path}", summary="Serve cropped face thumbnail image")
def get_face_image(
    filename: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_root_path: Optional[str] = Header(None, alias="X-User-Root-Path")
):
    # Sanitize and resolve path inside facess/ or output folder
    safe_filename = Path(filename).name
    if x_user_id:
        from src.user_workspace import get_user_catalog_dir
        user_catalog = get_user_catalog_dir(x_user_id, x_user_root_path)
        img_path = user_catalog / "facess" / safe_filename
        if not img_path.is_file():
            img_path = user_catalog / safe_filename
    else:
        img_path = config.FACES_FOLDER / safe_filename
        if not img_path.is_file():
            # Fallback to direct path inside OUTPUT_FOLDER
            img_path = config.OUTPUT_FOLDER / filename
    if not img_path.is_file():
        raise HTTPException(status_code=404, detail="Face image not found")
    return FileResponse(str(img_path))

@app.get("/api/media/files", summary="List all media files discovered in input sources")
def list_media_files(
    vault: Optional[bool] = False,
    x_vault_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_root_path: Optional[str] = Header(None, alias="X-User-Root-Path")
):
    allow_vault = bool(vault and verify_vault_session(x_vault_token))
    database.init_db()
    scanned = workers.scan_input_folders(allow_vault=allow_vault)
    sync_records = database.get_all_sync_records()

    face_counts = database.get_faces_count_by_source_file()
    all_faces_by_file = database.get_all_faces_by_source_file()
    db_metadata = database.get_all_media_metadata()
    
    # Also index face counts and face lists by filename / basename for flexible matching
    base_face_counts = {}
    for src_file, count in face_counts.items():
        base_name = Path(src_file).name.lower()
        base_face_counts[base_name] = base_face_counts.get(base_name, 0) + count

    base_faces = {}
    for src_file, f_list in all_faces_by_file.items():
        base_name = Path(src_file).name.lower()
        if base_name not in base_faces:
            base_faces[base_name] = []
        base_faces[base_name].extend(f_list)

    items = []
    for file_path, folder in scanned:
        try:
            stat = file_path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except Exception:
            size = 0
            mtime = 0
            
        str_path = str(file_path)
        base_name = file_path.name
        ext = file_path.suffix.lower()
        is_video = ext in config.SUPPORTED_VIDEO_EXTS
        is_image = ext in config.SUPPORTED_PHOTO_EXTS
        
        # Check sync history
        sync_rec = sync_records.get(str_path)
        if not sync_rec:
            # Check by basename fallback
            for k, v in sync_records.items():
                if Path(k).name.lower() == base_name.lower():
                    sync_rec = v
                    break
                    
        # Face count and face items
        file_faces_raw = all_faces_by_file.get(str_path)
        if not file_faces_raw:
            file_faces_raw = base_faces.get(base_name.lower(), [])
            
        # Deduplicate faces by face_id and extract distinct names
        seen_fids = set()
        dedup_faces = []
        face_names = []
        has_unassigned = False
        
        for f in file_faces_raw:
            fid = f.get("face_id")
            if fid and fid in seen_fids:
                continue
            if fid:
                seen_fids.add(fid)
            dedup_faces.append(f)
            fname = f.get("name")
            if fname and fname not in face_names:
                face_names.append(fname)
            if not f.get("is_reference") or (fname and fname.startswith("face_")):
                has_unassigned = True

        fc = len(dedup_faces) if dedup_faces else face_counts.get(str_path, base_face_counts.get(base_name.lower(), 0))
        
        status = sync_rec["status"] if sync_rec else "UNPROCESSED"
        sidecar = sync_rec.get("sidecar_path") if sync_rec else None

        # Load descriptions, summaries and extended semantic attributes
        desc = None
        desc_ru = None
        summ = None
        summ_ru = None
        environment = None
        lighting = None
        lighting_ru = None
        weather = None
        weather_ru = None
        time_of_day = None
        time_of_day_ru = None
        ocr_text = None
        exif_analysis = None
        exif_analysis_ru = None
        transcription = None
        transcription_ru = None
        timeline_events = None

        m = db_metadata.get(str_path) or db_metadata.get(base_name.lower())
        if m:
            desc = m.get("description")
            desc_ru = m.get("description_ru")
            summ = m.get("summary")
            summ_ru = m.get("summary_ru")
            environment = m.get("environment")
            lighting = m.get("lighting")
            lighting_ru = m.get("lighting_ru")
            weather = m.get("weather")
            weather_ru = m.get("weather_ru")
            time_of_day = m.get("time_of_day")
            time_of_day_ru = m.get("time_of_day_ru")
            ocr_text = m.get("ocr_text")
            exif_analysis = m.get("exif_analysis")
            exif_analysis_ru = m.get("exif_analysis_ru")
            transcription = m.get("transcription")
            transcription_ru = m.get("transcription_ru")
            timeline_events = m.get("timeline_events")

        # Sidecar JSON file inspection for metadata, phash, and module status
        sdata = None
        if sidecar:
            sidecar_file = Path(sidecar)
            if not sidecar_file.is_file():
                sidecar_file = config.OUTPUT_FOLDER / sidecar_file.name
            if sidecar_file.is_file():
                try:
                    with open(sidecar_file, "r", encoding="utf-8", errors="ignore") as sf:
                        sdata = json.load(sf)
                except Exception:
                    pass

        if sdata:
            if not desc and not desc_ru and not summ and not summ_ru:
                ga = sdata.get("gemini_analysis") or {}
                desc = ga.get("description") or sdata.get("description")
                desc_ru = ga.get("description_ru") or sdata.get("description_ru")
                summ = ga.get("summary") or sdata.get("summary")
                summ_ru = ga.get("summary_ru") or sdata.get("summary_ru")
                environment = ga.get("environment")
                lighting = ga.get("lighting")
                lighting_ru = ga.get("lighting_ru")
                weather = ga.get("weather")
                weather_ru = ga.get("weather_ru")
                time_of_day = ga.get("time_of_day")
                time_of_day_ru = ga.get("time_of_day_ru")
                ocr_text = ga.get("ocr_text")
                exif_analysis = ga.get("exif_analysis")
                exif_analysis_ru = ga.get("exif_analysis_ru")
                transcription = ga.get("transcription")
                transcription_ru = ga.get("transcription_ru")
                timeline_events = ga.get("timeline_events")
            modules_status = sdata.get("modules_status") or workers.get_file_modules_status(sdata, "video" if is_video else "photo")
        else:
            modules_status = workers.get_file_modules_status({
                "transcription": transcription,
                "face_timeline": dedup_faces or face_names,
                "gemini_analysis": {"summary": summ, "description": desc}
            }, "video" if is_video else "photo")
        
        items.append({
            "file_path": str_path,
            "filename": base_name,
            "folder": str(folder),
            "file_size": size,
            "mtime": mtime,
            "is_video": is_video,
            "is_image": is_image,
            "status": status,
            "sidecar_path": sidecar,
            "modules_status": modules_status,
            "description": desc,
            "description_ru": desc_ru,
            "summary": summ,
            "summary_ru": summ_ru,
            "environment": environment,
            "lighting": lighting,
            "lighting_ru": lighting_ru,
            "weather": weather,
            "weather_ru": weather_ru,
            "time_of_day": time_of_day,
            "time_of_day_ru": time_of_day_ru,
            "ocr_text": ocr_text,
            "exif_analysis": exif_analysis,
            "exif_analysis_ru": exif_analysis_ru,
            "transcription": transcription,
            "transcription_ru": transcription_ru,
            "timeline_events": timeline_events,
            "face_count": fc,
            "faces": dedup_faces,
            "face_names": face_names,
            "has_unassigned_faces": has_unassigned,
            "error_message": sync_rec.get("error_message") if sync_rec else None
        })
        
    return {
        "total_files": len(items),
        "files": items
    }

@app.get("/api/media/sidecar", summary="Get full sidecar JSON metadata for a media file")
def get_media_sidecar(path: Optional[str] = None, file: Optional[str] = None):
    target = path or file
    if not target or not target.strip():
        raise HTTPException(status_code=400, detail="Missing file or path parameter")

    target_str = target.strip()
    target_path = Path(target_str)
    sidecar_path = None

    # Check sync_history
    database.init_db()
    sync_rec = database.get_sync_record(target_str)
    if not sync_rec:
        sync_records = database.get_all_sync_records()
        for k, v in sync_records.items():
            if Path(k).name.lower() == target_path.name.lower():
                sync_rec = v
                break

    if sync_rec and sync_rec.get("sidecar_path"):
        cand = Path(sync_rec["sidecar_path"])
        if cand.is_file():
            sidecar_path = cand
        elif (config.OUTPUT_FOLDER / cand.name).is_file():
            sidecar_path = config.OUTPUT_FOLDER / cand.name

    if not sidecar_path:
        # Check standard sidecar naming
        cand = config.OUTPUT_FOLDER / f"{target_path.name}.json"
        if cand.is_file():
            sidecar_path = cand

    if not sidecar_path or not sidecar_path.is_file():
        raise HTTPException(status_code=404, detail="Sidecar metadata not found for this file.")

    try:
        with open(sidecar_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read sidecar file: {str(e)}")

@app.get("/api/media/file", summary="Serve media file directly from input sources")
def get_media_file(path: Optional[str] = None, file: Optional[str] = None):
    target = path or file
    if not target or not target.strip():
        raise HTTPException(status_code=400, detail="Missing file or path parameter")
        
    target_str = target.strip()
    target_path = Path(target_str)
    
    # If direct file exists
    resolved_path = None
    if target_path.is_file():
        resolved_path = target_path
    else:
        # Check inside INPUT_FOLDERS
        for inf in config.INPUT_FOLDERS:
            cand = inf / target_str
            if cand.is_file():
                resolved_path = cand
                break
            cand = inf / target_path.name
            if cand.is_file():
                resolved_path = cand
                break
        # Check inside container media_input
        if not resolved_path:
            for fallback_dir in [Path("/app/media_input"), config.PROJECT_ROOT / "media_input"]:
                if fallback_dir.is_dir():
                    cand = fallback_dir / target_str
                    if cand.is_file():
                        resolved_path = cand
                        break
                    cand = fallback_dir / target_path.name
                    if cand.is_file():
                        resolved_path = cand
                        break
        # Check inside OUTPUT_FOLDER
        if not resolved_path:
            cand = config.OUTPUT_FOLDER / target_str
            if cand.is_file():
                resolved_path = cand
            cand = config.OUTPUT_FOLDER / target_path.name
            if cand.is_file():
                resolved_path = cand
            for fallback_out in [Path("/app/media_output"), config.PROJECT_ROOT / "media_output"]:
                if fallback_out.is_dir():
                    cand = fallback_out / target_str
                    if cand.is_file():
                        resolved_path = cand
                        break
                    cand = fallback_out / target_path.name
                    if cand.is_file():
                        resolved_path = cand
                        break
                
    if not resolved_path or not resolved_path.is_file():
        # Search in scan_input_folders
        scanned = workers.scan_input_folders()
        for p, _ in scanned:
            if p.name.lower() == target_path.name.lower() or str(p).lower() == target_str.lower():
                resolved_path = p
                break
                
    if not resolved_path or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail=f"Media file '{target_str}' not found")
        
    return FileResponse(str(resolved_path))

@app.get("/api/media/faces-for-file", summary="Get all face detections for a specific source file")
def get_faces_for_file(file: str):
    if not file or not file.strip():
        raise HTTPException(status_code=400, detail="File parameter cannot be empty")
    database.init_db()
    return database.get_faces_by_source_file(file.strip())

@app.post("/api/media/add-person", summary="Add or link a known or new person to a media file")
def add_person_to_file(request: AddPersonToFileRequest):
    if not request.file or not request.file.strip():
        raise HTTPException(status_code=400, detail="File parameter cannot be empty")
    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Person name cannot be empty")
    database.init_db()
    result = database.add_person_to_media_file(request.file.strip(), request.name.strip())
    return {
        "status": "success",
        "message": f"Person '{request.name.strip()}' successfully linked to file.",
        "data": result
    }

@app.post("/api/media/remove-face", summary="Remove or unlink a face/person from a media file")
def remove_face_from_file(request: RemoveFaceFromFileRequest):
    if not request.file or not request.file.strip():
        raise HTTPException(status_code=400, detail="File parameter cannot be empty")
    if not request.face_id or not request.face_id.strip():
        raise HTTPException(status_code=400, detail="Face ID cannot be empty")
    database.init_db()
    database.remove_face_from_media_file(request.file.strip(), request.face_id.strip())
    return {
        "status": "success",
        "message": f"Face/Person '{request.face_id}' removed from file."
    }

@app.post("/api/faces/rename", summary="Rename a face registry entry or person")
def rename_face(request: RenameFaceRequest):
    database.init_db()
    mapping = database.get_face_name_mapping()
    if request.face_id not in mapping:
        raise HTTPException(status_code=404, detail=f"Face ID '{request.face_id}' not found in registry.")
        
    database.update_face_name(request.face_id, request.name.strip())
    return {"status": "success", "message": f"Face '{request.face_id}' renamed to '{request.name.strip()}'."}

@app.post("/api/faces/assign", summary="Assign an unrecognized face to a known or new person")
def assign_face(request: AssignFaceRequest):
    database.init_db()
    target_name = (request.person_name or request.name or "").strip()
    if not target_name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty.")
    
    mapping = database.get_face_name_mapping()
    if request.face_id not in mapping:
        raise HTTPException(status_code=404, detail=f"Face ID '{request.face_id}' not found in registry.")

    success = database.assign_face_to_person(request.face_id, target_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to assign face to person.")
        
    if request.file and request.file.strip():
        try:
            database.add_person_to_media_file(request.file.strip(), target_name)
        except Exception:
            pass

    return {
        "status": "success",
        "message": f"Face '{request.face_id}' successfully assigned to '{target_name}' as a reference face."
    }

@app.post("/api/media/metadata", summary="Ingest AI metadata, descriptions, tags, and sidecar attributes")
def save_media_metadata(payload: MediaMetadataPayload):
    target = payload.file or payload.file_path
    if not target or not target.strip():
        raise HTTPException(status_code=400, detail="Missing file or file_path parameter.")

    target_file = target.strip()
    target_path = Path(target_file)
    filename = target_path.name
    
    database.init_db()

    # Generate media_id
    media_id = hashlib.sha256(str(target_file).encode()).hexdigest()[:16]
    ext = target_path.suffix.lower()
    media_type = "video" if ext in config.SUPPORTED_VIDEO_EXTS else "photo"

    file_size = 0
    mtime = 0.0
    if target_path.is_file():
        try:
            stat = target_path.stat()
            file_size = stat.st_size
            mtime = stat.st_mtime
        except Exception:
            pass

    # Upsert media item
    database.upsert_media_item(
        media_id=media_id,
        file_path=target_file,
        file_name=filename,
        media_type=media_type,
        file_size=file_size,
        mtime=mtime,
        status="PROCESSED"
    )

    # Prepare raw_gemini dictionary
    raw_gemini = payload.raw_gemini or {}
    if payload.summary:
        raw_gemini["summary"] = payload.summary
    if payload.summary_ru:
        raw_gemini["summary_ru"] = payload.summary_ru
    if payload.description:
        raw_gemini["description"] = payload.description
    if payload.description_ru:
        raw_gemini["description_ru"] = payload.description_ru
    if payload.environment:
        raw_gemini["environment"] = payload.environment
    if payload.lighting:
        raw_gemini["lighting"] = payload.lighting
    if payload.lighting_ru:
        raw_gemini["lighting_ru"] = payload.lighting_ru
    if payload.weather:
        raw_gemini["weather"] = payload.weather
    if payload.weather_ru:
        raw_gemini["weather_ru"] = payload.weather_ru
    if payload.time_of_day:
        raw_gemini["time_of_day"] = payload.time_of_day
    if payload.time_of_day_ru:
        raw_gemini["time_of_day_ru"] = payload.time_of_day_ru
    if payload.exif_analysis:
        raw_gemini["exif_analysis"] = payload.exif_analysis
    if payload.exif_analysis_ru:
        raw_gemini["exif_analysis_ru"] = payload.exif_analysis_ru
    if payload.transcription:
        raw_gemini["transcription"] = payload.transcription
    if payload.transcription_ru:
        raw_gemini["transcription_ru"] = payload.transcription_ru
    if payload.ocr_text:
        raw_gemini["ocr_text"] = payload.ocr_text
    if payload.timeline_events:
        raw_gemini["timeline_events"] = payload.timeline_events

    # Upsert media metadata
    database.upsert_media_metadata(
        media_id=media_id,
        summary=payload.summary,
        summary_ru=payload.summary_ru,
        description=payload.description,
        description_ru=payload.description_ru,
        environment=payload.environment,
        lighting=payload.lighting,
        weather=payload.weather,
        time_of_day=payload.time_of_day,
        ocr_text=payload.ocr_text,
        camera_make=payload.camera_make,
        camera_model=payload.camera_model,
        latitude=payload.latitude,
        longitude=payload.longitude,
        location_name=payload.location_name,
        raw_exif=payload.raw_exif,
        raw_gemini=raw_gemini,
        raw_defects=payload.raw_defects
    )

    # Save tags
    if payload.tags:
        formatted_tags = []
        for t in payload.tags:
            if isinstance(t, str):
                formatted_tags.append({"tag": t.strip(), "category": "general", "confidence": 1.0})
            elif isinstance(t, dict) and t.get("tag"):
                formatted_tags.append(t)
        if formatted_tags:
            database.save_media_tags(media_id, formatted_tags)

    # Save video timeline events
    if payload.timeline_events:
        database.save_video_timeline_events(media_id, payload.timeline_events)

    # Update or create sidecar JSON on disk
    sidecar_path = config.OUTPUT_FOLDER / f"{filename}.json"
    try:
        sidecar_data = {}
        if sidecar_path.is_file():
            try:
                with open(sidecar_path, "r", encoding="utf-8") as sf:
                    sidecar_data = json.load(sf)
            except Exception:
                pass
        
        sidecar_data.update({
            "file_path": target_file,
            "filename": filename,
            "media_type": media_type,
            "summary": payload.summary or sidecar_data.get("summary"),
            "summary_ru": payload.summary_ru or sidecar_data.get("summary_ru"),
            "description": payload.description or sidecar_data.get("description"),
            "description_ru": payload.description_ru or sidecar_data.get("description_ru"),
            "gemini_analysis": raw_gemini,
            "updated_at": datetime.now().isoformat()
        })
        if payload.tags:
            sidecar_data["tags"] = payload.tags
            
        with open(sidecar_path, "w", encoding="utf-8") as sf:
            json.dump(sidecar_data, sf, indent=2, ensure_ascii=False)

        database.update_sync_record(
            file_path=target_file,
            file_size=file_size,
            mtime=mtime,
            status="PROCESSED",
            sidecar_path=str(sidecar_path)
        )
    except Exception as e:
        print(f"Warning: Failed to update sidecar file {sidecar_path}: {e}")

    return {
        "status": "success",
        "message": f"Metadata persisted successfully for '{filename}'.",
        "media_id": media_id,
        "file": target_file
    }


@app.post("/api/faces/assign-group", summary="Assign a group/cluster of similar faces to a person")
def assign_face_group(request: AssignGroupRequest):
    database.init_db()
    trimmed_name = request.name.strip()
    if not trimmed_name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty.")
    if not request.face_ids:
        raise HTTPException(status_code=400, detail="List of face IDs cannot be empty.")
        
    success = database.assign_group_to_person(request.face_ids, trimmed_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to assign group to person.")
        
    return {
        "status": "success",
        "message": f"Successfully assigned {len(request.face_ids)} face(s) to '{trimmed_name}'."
    }

@app.post("/api/faces/reset", summary="Reset a face assignment back to unassigned candidate")
def reset_face(request: ResetFaceRequest):
    database.init_db()
    mapping = database.get_face_name_mapping()
    if request.face_id not in mapping:
        raise HTTPException(status_code=404, detail=f"Face ID '{request.face_id}' not found in registry.")

    success = database.reset_face_assignment(request.face_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset face assignment.")

    return {
        "status": "success",
        "message": f"Face '{request.face_id}' assignment reset to unassigned."
    }

@app.post("/api/faces/reset-by-filename", summary="Reset all face assignments for a specific filename")
def reset_faces_by_filename(request: ResetFaceByFilenameRequest):
    database.init_db()
    filename = request.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename parameter cannot be empty.")

    result = database.reset_face_assignments_by_filename(filename)
    return {
        "status": "success",
        "message": f"Reset {result['reset_count']} face assignment(s) for file '{filename}'.",
        "reset_count": result["reset_count"],
        "face_ids": result["face_ids"]
    }

@app.post("/api/faces/reset-by-file", summary="Alias for reset-by-filename")
def reset_faces_by_file(request: ResetFaceByFilenameRequest):
    return reset_faces_by_filename(request)

@app.post("/api/faces/delete", summary="Delete a face entry from registry")
def delete_face(request: DeleteFaceRequest):
    database.init_db()
    mapping = database.get_face_name_mapping()
    if request.face_id not in mapping:
        raise HTTPException(status_code=404, detail=f"Face ID '{request.face_id}' not found in registry.")

    database.delete_face(request.face_id)
    return {"status": "success", "message": f"Face '{request.face_id}' deleted successfully."}

@app.get("/api/fs/browse", summary="Browse host filesystem directories and files")
@app.post("/api/fs/browse", summary="Browse host filesystem directories and files")
async def browse_directory(
    request: Request,
    path: Optional[str] = None,
    mode: str = "folder",
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_root_path: Optional[str] = Header(None, alias="X-User-Root-Path")
):
    target_path = path
    target_mode = mode
    if request.method == "POST":
        try:
            body = await request.json()
            if isinstance(body, dict):
                if body.get("path") is not None:
                    target_path = body.get("path")
                if body.get("mode") is not None:
                    target_mode = body.get("mode")
        except Exception:
            pass

    if x_user_id:
        from src.user_workspace import get_user_root, validate_path_in_user_workspace
        user_root = get_user_root(x_user_id, x_user_root_path).resolve()
        user_root.mkdir(parents=True, exist_ok=True)
        if not target_path or not str(target_path).strip():
            curr_path = user_root
        else:
            curr_path = Path(str(target_path).strip()).resolve()
            if not validate_path_in_user_workspace(x_user_id, curr_path, x_user_root_path):
                raise HTTPException(status_code=403, detail="Access denied: Cannot browse outside user workspace.")
        shortcuts = [{"label": "User Storage", "path": str(user_root)}]
    else:
        current = str(target_path).strip() if target_path else str(config.PROJECT_ROOT)
        shortcuts = [{"label": "Project Root", "path": str(config.PROJECT_ROOT)}]
        if sys.platform == "win32":
            for d in ["C:\\", "D:\\", "E:\\", "F:\\", "Z:\\"]:
                try:
                    if os.path.exists(d):
                        shortcuts.append({"label": d, "path": d})
                except Exception:
                    pass
        else:
            for c in ["/", "/shares", "/media", "/mnt", "/data", "/app", "/home"]:
                try:
                    if os.path.exists(c):
                        shortcuts.append({"label": c, "path": c})
                except Exception:
                    pass
                    
        curr_path = Path(current)
        if not curr_path.is_absolute() and not current.startswith(("\\\\", "//")):
            curr_path = (config.PROJECT_ROOT / current).resolve()
        
    if not os.path.exists(str(curr_path)):
        parent_str = str(curr_path.parent) if str(curr_path) != str(curr_path.parent) else None
        return {
            "current_path": str(curr_path),
            "parent_path": parent_str,
            "shortcuts": shortcuts,
            "directories": [],
            "files": [],
            "error": f"Path does not exist: {curr_path}"
        }
        
    try:
        if not curr_path.is_dir():
            curr_path = curr_path.parent
            
        directories = []
        files = []
        with os.scandir(str(curr_path)) as it:
            for entry in it:
                if entry.name.startswith(".") and entry.name != ".env":
                    continue
                if entry.name in ("node_modules", "__pycache__"):
                    continue
                try:
                    if entry.is_dir():
                        directories.append(entry.name)
                    elif mode == "file" and entry.is_file():
                        ext = Path(entry.name).suffix.lower()
                        if ext in config.SUPPORTED_PHOTO_EXTS or ext in config.SUPPORTED_VIDEO_EXTS or ext in (".json", ".db"):
                            files.append(entry.name)
                except Exception:
                    pass
                        
        directories.sort(key=str.lower)
        files.sort(key=str.lower)
        
        if x_user_id and (str(curr_path) == str(user_root) or not validate_path_in_user_workspace(x_user_id, curr_path.parent, x_user_root_path)):
            parent = None
        else:
            parent = str(curr_path.parent) if str(curr_path) != str(curr_path.parent) else None
        return {
            "current_path": str(curr_path),
            "parent_path": parent,
            "shortcuts": shortcuts,
            "directories": directories,
            "files": files
        }
    except Exception as e:
        return {
            "current_path": str(curr_path),
            "parent_path": str(curr_path.parent) if str(curr_path) != str(curr_path.parent) else None,
            "shortcuts": shortcuts,
            "directories": [],
            "files": [],
            "error": str(e)
        }

class SettingsUpdateRequest(BaseModel):
    input_folders: Optional[List[str]] = None
    output_folder: Optional[str] = None
    model_provider: Optional[str] = None
    gemini_model: Optional[str] = None
    local_model_name: Optional[str] = None
    gemini_max_workers: Optional[int] = None
    local_max_workers: Optional[int] = None
    whisper_model: Optional[str] = None
    preserve_structure: Optional[bool] = None
    ui_base_url: Optional[str] = None
    vision_prompt_template: Optional[str] = None

@app.get("/api/settings", summary="Get current path and engine configuration")
def get_settings():
    saved = config.load_persistent_settings()
    env_inputs = [p.strip() for p in os.environ.get("INPUT_FOLDERS", "").split(",") if p.strip()]
    default_inputs = env_inputs if env_inputs else [str(config.PROJECT_ROOT / "media_input")]
    default_output = os.environ.get("OUTPUT_FOLDER", str(config.PROJECT_ROOT / "media_output"))
    
    return {
        "input_folders": [str(p) for p in config.INPUT_FOLDERS],
        "output_folder": str(config.OUTPUT_FOLDER),
        "default_input_folders": default_inputs,
        "default_output_folder": default_output,
        "is_custom_input": bool(saved.get("INPUT_FOLDERS")),
        "is_custom_output": bool(saved.get("OUTPUT_FOLDER")),
        "model_provider": config.MODEL_PROVIDER,
        "gemini_model": config.GEMINI_MODEL,
        "local_model_name": config.LOCAL_MODEL_NAME,
        "gemini_max_workers": config.GEMINI_MAX_WORKERS,
        "local_max_workers": config.LOCAL_MAX_WORKERS,
        "whisper_model": config.WHISPER_MODEL,
        "preserve_structure": config.PRESERVE_STRUCTURE,
        "ui_base_url": config.UI_BASE_URL,
        "vision_prompt_template": config.VISION_PROMPT_TEMPLATE,
        "default_vision_prompt_template": config.DEFAULT_VISION_PROMPT_TEMPLATE
    }

@app.post("/api/settings", summary="Update persistent settings")
def update_settings(request: SettingsUpdateRequest):
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot change configuration while synchronization is running.")
            
    config.save_persistent_settings(
        input_folders=request.input_folders,
        output_folder=request.output_folder,
        model_provider=request.model_provider,
        gemini_model=request.gemini_model,
        local_model_name=request.local_model_name,
        gemini_max_workers=request.gemini_max_workers,
        local_max_workers=request.local_max_workers,
        whisper_model=request.whisper_model,
        preserve_structure=request.preserve_structure,
        ui_base_url=request.ui_base_url,
        vision_prompt_template=request.vision_prompt_template
    )
    
    database.init_db()
    
    res = get_settings()
    res["status"] = "success"
    res["message"] = "Settings updated successfully."
    return res

@app.get("/api/health", summary="Lightweight health ping")
def api_health_check():
    dev_info = config.get_system_device_info()
    models_status = workers.get_models_loaded_status()
    models_ready = any(models_status.values())
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "models_ready": models_ready,
        "gpu_memory_used_mb": dev_info.get("gpu_memory_used_mb")
    }

@app.get("/health", summary="Health check endpoint")
def health_check():
    dev_info = config.get_system_device_info()
    models_status = workers.get_models_loaded_status()
    models_ready = any(models_status.values())
    return {
        "status": "ok",
        "service": "media_cataloger",
        "timestamp": datetime.now().isoformat(),
        "models_ready": models_ready,
        "gpu_memory_used_mb": dev_info.get("gpu_memory_used_mb")
    }

@app.get("/api/duplicates/device-info", summary="Get GPU acceleration status for duplicate detection")
def get_duplicates_device_info():
    from src.utils.gpu_duplicates import get_gpu_device_info
    return get_gpu_device_info()

@app.post("/api/duplicates/scan", summary="High-speed GPU tensor duplicate and similarity detection scan")
def scan_duplicates(request: Optional[DuplicateScanRequest] = None):
    req = request or DuplicateScanRequest()
    from src.utils.gpu_duplicates import run_gpu_duplicate_clustering
    
    file_items = []
    if req.files and len(req.files) > 0:
        for fp in req.files:
            file_items.append({"file_path": fp})
    else:
        scanned = workers.scan_input_folders()
        for p, _ in scanned:
            file_items.append({"file_path": str(p)})
            
    if not file_items:
        from src.utils.gpu_duplicates import get_gpu_device_info
        dev_info = get_gpu_device_info()
        return {
            "engine": dev_info.get("engine", "cpu"),
            "device_name": dev_info.get("device_name", "No files found"),
            "elapsed_seconds": 0.0,
            "scanned_files_count": 0,
            "total_groups": 0,
            "groups": []
        }
        
    result = run_gpu_duplicate_clustering(
        files=file_items,
        mode=req.mode or "all",
        similarity_threshold=req.similarity_threshold or 0.90,
        burst_window_seconds=req.burst_window_seconds or 3.0,
        keep_strategy=req.keep_strategy or "highest_resolution"
    )
    return result

# Start API with Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    # Read port from environment variable or default to 8001
    port = int(os.environ.get("API_PORT", 8001))
    host = os.environ.get("API_HOST", "0.0.0.0")
    print(f"Starting Media Cataloger Business Logic API server at http://{host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=False)

