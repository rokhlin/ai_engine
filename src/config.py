import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union


# Simple function to load .env file if it exists
def load_dotenv(dotenv_path: Path):
    if dotenv_path.is_file():
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION: str = "2.4.0"

def get_system_device_info() -> Dict[str, Any]:
    """Retrieve compute device name and memory allocation metrics."""
    device = "cpu"
    gpu_memory_used_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            gpu_memory_used_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        pass
    return {
        "device": device,
        "gpu_memory_used_mb": gpu_memory_used_mb
    }

custom_env = os.environ.get("ENV_FILE") or os.environ.get("ENVIRONMENT_FILE")
if custom_env:
    custom_path = Path(custom_env)
    if not custom_path.is_absolute():
        custom_path = PROJECT_ROOT / custom_path
    load_dotenv(custom_path)
else:
    # Load .env first if present (default to data/config/.env, fallback to root .env)
    config_dir = PROJECT_ROOT / "data" / "config"
    if (config_dir / ".env").is_file():
        load_dotenv(config_dir / ".env")
    elif (PROJECT_ROOT / ".env").is_file():
        load_dotenv(PROJECT_ROOT / ".env")

    # Layer .env.local on top if present
    if (config_dir / ".env.local").is_file():
        load_dotenv(config_dir / ".env.local")
    elif (PROJECT_ROOT / ".env.local").is_file():
        load_dotenv(PROJECT_ROOT / ".env.local")

# --- Persistent Settings Support ---
def get_settings_file_path() -> Path:
    custom_cfg = os.environ.get("CONFIG_PATH")
    if custom_cfg and custom_cfg.strip():
        return Path(custom_cfg.strip()) / "settings.json"
    custom_set = os.environ.get("SETTINGS_PATH")
    if custom_set and custom_set.strip():
        return Path(custom_set.strip())
    config_dir_settings = PROJECT_ROOT / "data" / "config" / "settings.json"
    if config_dir_settings.is_file():
        return config_dir_settings
    return PROJECT_ROOT / "settings.json"

def load_persistent_settings() -> dict:
    settings = {}
    settings_path = get_settings_file_path()
    if settings_path.is_file():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception:
            pass
    return settings

def save_persistent_settings(
    input_folders: Optional[List[str]] = None,
    output_folder: Optional[str] = None,
    model_provider: Optional[str] = None,
    gemini_model: Optional[str] = None,
    local_model_name: Optional[str] = None,
    gemini_max_workers: Optional[int] = None,
    local_max_workers: Optional[int] = None,
    whisper_model: Optional[str] = None,
    preserve_structure: Optional[bool] = None,
    ui_base_url: Optional[str] = None,
    vision_prompt_template: Optional[str] = None,
    **extra_kwargs
):
    global INPUT_FOLDERS, OUTPUT_FOLDER, DB_PATH, FACES_FOLDER
    global MODEL_PROVIDER, GEMINI_MODEL, LOCAL_MODEL_NAME
    global GEMINI_MAX_WORKERS, LOCAL_MAX_WORKERS, WHISPER_MODEL, PRESERVE_STRUCTURE, UI_BASE_URL
    global VISION_PROMPT_TEMPLATE
    
    settings_path = get_settings_file_path()
    data = load_persistent_settings()
    
    # Clean paths if provided
    if input_folders is not None:
        input_paths = [Path(p.strip()) for p in input_folders if p.strip()]
        if input_paths:
            data["INPUT_FOLDERS"] = [str(p) for p in input_paths]
            INPUT_FOLDERS = input_paths
        else:
            data.pop("INPUT_FOLDERS", None)
            env_inputs = [Path(p.strip()) for p in os.environ.get("INPUT_FOLDERS", "").split(",") if p.strip()]
            INPUT_FOLDERS = env_inputs if env_inputs else [PROJECT_ROOT / "media_input"]
            
    if output_folder is not None:
        output_path_str = output_folder.strip()
        if output_path_str:
            data["OUTPUT_FOLDER"] = output_path_str
            OUTPUT_FOLDER = Path(output_path_str)
            if not os.environ.get("DB_PATH"):
                DB_PATH = OUTPUT_FOLDER / "catalog_history.db"
            FACES_FOLDER = OUTPUT_FOLDER / "facess"
        else:
            data.pop("OUTPUT_FOLDER", None)
            OUTPUT_FOLDER = Path(os.environ.get("OUTPUT_FOLDER", str(PROJECT_ROOT / "media_output")))
            if not os.environ.get("DB_PATH"):
                DB_PATH = OUTPUT_FOLDER / "catalog_history.db"
            FACES_FOLDER = OUTPUT_FOLDER / "facess"
            
    if model_provider is not None:
        if model_provider.strip():
            MODEL_PROVIDER = model_provider.strip().lower()
            data["MODEL_PROVIDER"] = MODEL_PROVIDER
        else:
            data.pop("MODEL_PROVIDER", None)
            MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini").lower()
            
    # Add support for USERS_BASE_DIR in multi-tenant environment
    global USERS_BASE_DIR
    USERS_BASE_DIR = Path(os.environ.get("USERS_BASE_DIR", "/app/storage/users"))

        
    if gemini_model is not None:
        if gemini_model.strip():
            GEMINI_MODEL = gemini_model.strip()
            data["GEMINI_MODEL"] = GEMINI_MODEL
        else:
            data.pop("GEMINI_MODEL", None)
            GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        
    if local_model_name is not None:
        if local_model_name.strip():
            LOCAL_MODEL_NAME = local_model_name.strip()
            data["LOCAL_MODEL_NAME"] = LOCAL_MODEL_NAME
        else:
            data.pop("LOCAL_MODEL_NAME", None)
            LOCAL_MODEL_NAME = os.environ.get("LOCAL_MODEL_NAME", "")
        
    if gemini_max_workers is not None:
        GEMINI_MAX_WORKERS = max(1, int(gemini_max_workers))
        data["GEMINI_MAX_WORKERS"] = GEMINI_MAX_WORKERS
        
    if local_max_workers is not None:
        LOCAL_MAX_WORKERS = max(1, int(local_max_workers))
        data["LOCAL_MAX_WORKERS"] = LOCAL_MAX_WORKERS
        
    if whisper_model is not None:
        if whisper_model.strip():
            WHISPER_MODEL = whisper_model.strip()
            data["WHISPER_MODEL"] = WHISPER_MODEL
        else:
            data.pop("WHISPER_MODEL", None)
            WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
        
    if preserve_structure is not None:
        PRESERVE_STRUCTURE = bool(preserve_structure)
        data["PRESERVE_STRUCTURE"] = PRESERVE_STRUCTURE

    if ui_base_url is not None:
        if ui_base_url.strip():
            UI_BASE_URL = ui_base_url.strip()
            data["UI_BASE_URL"] = UI_BASE_URL
        else:
            data.pop("UI_BASE_URL", None)
            UI_BASE_URL = os.environ.get("UI_BASE_URL", "http://localhost:8000")

    if vision_prompt_template is not None:
        if vision_prompt_template.strip():
            VISION_PROMPT_TEMPLATE = vision_prompt_template.strip()
            data["VISION_PROMPT_TEMPLATE"] = VISION_PROMPT_TEMPLATE
        else:
            data.pop("VISION_PROMPT_TEMPLATE", None)
            VISION_PROMPT_TEMPLATE = DEFAULT_VISION_PROMPT_TEMPLATE

    for k, v in extra_kwargs.items():
        if v is not None:
            data[k] = v

    if data:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        if settings_path.is_file():
            settings_path.unlink()
            
    # Ensure directories exist if accessible
    try:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        FACES_FOLDER.mkdir(parents=True, exist_ok=True)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def decrypt_secret(val: Union[str, Any]) -> str:
    """Decrypt AES-256-GCM secret prefixed with enc:v1: if cryptography is available, else return as-is."""
    if not val or not isinstance(val, str) or not val.strip().startswith("enc:v1:"):
        return str(val).strip() if val else ""
    try:
        import hashlib
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        parts = val.strip()[len("enc:v1:"):].split(":")
        if len(parts) != 3:
            return str(val).strip()
        iv_hex, tag_hex, ct_hex = parts
        secret = os.environ.get("APP_SECRET_KEY") or os.environ.get("JWT_SECRET") or "media_cataloger_secure_vault_secret_2026"
        salt = os.environ.get("ENCRYPTION_SALT", "media_cataloger_salt_v1")
        key = hashlib.scrypt(secret.encode("utf-8"), salt=salt.encode("utf-8"), n=16384, r=8, p=1, maxmem=0, dklen=32)
        aesgcm = AESGCM(key)
        data = bytes.fromhex(ct_hex) + bytes.fromhex(tag_hex)
        decrypted = aesgcm.decrypt(bytes.fromhex(iv_hex), data, None)
        return decrypted.decode("utf-8")
    except Exception:
        return str(val).strip()

def is_running_in_docker() -> bool:
    """Detect if current process is running inside a Docker container."""
    return os.path.isfile("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER", "").lower() in ("true", "1")

def normalize_container_path(folder: Union[str, Path], fallback_type: str = "input") -> Path:
    """Normalize and resolve folder paths, handling Windows drive and UNC paths inside Docker containers."""
    folder_str = str(folder).strip()
    if is_running_in_docker():
        if not folder_str.startswith("/") or ":" in folder_str or folder_str.startswith(("\\\\", "//")):
            lower = folder_str.lower()
            if fallback_type == "output" or any(k in lower for k in ("output", "cataloger", "sda1")):
                cand = Path("/app/media_output")
                if cand.is_dir():
                    return cand
                cand2 = PROJECT_ROOT / "media_output"
                if cand2.is_dir():
                    return cand2
                return cand
            else:
                cand = Path("/app/media_input")
                if cand.is_dir():
                    return cand
                cand2 = PROJECT_ROOT / "media_input"
                if cand2.is_dir():
                    return cand2
                return cand
    p = Path(folder_str)
    return p

def resolve_target_output_folder(folder: Union[str, Path]) -> Path:
    """Normalize and resolve output folder, handling Windows drive paths inside Docker containers."""
    return normalize_container_path(folder, fallback_type="output")

def apply_runtime_settings(
    output_folder: Optional[str] = None,
    model_provider: Optional[str] = None,
    gemini_model: Optional[str] = None,
    local_model_name: Optional[str] = None,
    gemini_max_workers: Optional[int] = None,
    local_max_workers: Optional[int] = None,
    whisper_model: Optional[str] = None,
    preserve_structure: Optional[bool] = None,
    ui_base_url: Optional[str] = None,
    target_tags: Optional[List[str]] = None,
    tag_format: Optional[str] = None,
    vision_prompt_template: Optional[str] = None,
    **kwargs
):
    """Apply dynamic in-memory settings passed in runtime task requests without writing to disk."""
    global OUTPUT_FOLDER, DB_PATH, FACES_FOLDER
    global MODEL_PROVIDER, GEMINI_MODEL, LOCAL_MODEL_NAME, GEMINI_API_KEY
    global GEMINI_MAX_WORKERS, LOCAL_MAX_WORKERS, WHISPER_MODEL, PRESERVE_STRUCTURE, UI_BASE_URL
    global VISION_PROMPT_TEMPLATE

    if output_folder and str(output_folder).strip():
        OUTPUT_FOLDER = resolve_target_output_folder(str(output_folder).strip())
        DB_PATH = OUTPUT_FOLDER / "catalog_history.db"
        FACES_FOLDER = OUTPUT_FOLDER / "facess"
        try:
            OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
            FACES_FOLDER.mkdir(parents=True, exist_ok=True)
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    if model_provider and str(model_provider).strip():
        MODEL_PROVIDER = str(model_provider).strip().lower()

    if gemini_model and str(gemini_model).strip():
        GEMINI_MODEL = str(gemini_model).strip()

    gemini_key_arg = kwargs.get("gemini_api_key")
    if gemini_key_arg and str(gemini_key_arg).strip():
        GEMINI_API_KEY = decrypt_secret(str(gemini_key_arg).strip())
        try:
            from src.utils import gemini
            with gemini._client_lock:
                gemini._client = None
        except Exception:
            pass

    if local_model_name and str(local_model_name).strip():
        LOCAL_MODEL_NAME = str(local_model_name).strip()

    if gemini_max_workers is not None:
        GEMINI_MAX_WORKERS = max(1, int(gemini_max_workers))

    if local_max_workers is not None:
        LOCAL_MAX_WORKERS = max(1, int(local_max_workers))

    if whisper_model and str(whisper_model).strip():
        WHISPER_MODEL = str(whisper_model).strip()

    if preserve_structure is not None:
        PRESERVE_STRUCTURE = bool(preserve_structure)

    global TARGET_TAGS, TAG_FORMAT

    if target_tags is not None:
        if isinstance(target_tags, list):
            TARGET_TAGS = [str(t).strip() for t in target_tags if str(t).strip()]
        elif isinstance(target_tags, str):
            TARGET_TAGS = [str(t).strip() for t in target_tags.split(",") if str(t).strip()]
        else:
            TARGET_TAGS = None
    elif "target_tags" in kwargs and kwargs["target_tags"] is not None:
        tt = kwargs["target_tags"]
        if isinstance(tt, list):
            TARGET_TAGS = [str(t).strip() for t in tt if str(t).strip()]
        elif isinstance(tt, str):
            TARGET_TAGS = [str(t).strip() for t in tt.split(",") if str(t).strip()]

    if tag_format is not None and str(tag_format).strip():
        TAG_FORMAT = str(tag_format).strip()
    elif "tag_format" in kwargs and kwargs["tag_format"] is not None:
        TAG_FORMAT = str(kwargs["tag_format"]).strip()

    if vision_prompt_template is not None:
        if str(vision_prompt_template).strip():
            VISION_PROMPT_TEMPLATE = str(vision_prompt_template).strip()
        else:
            VISION_PROMPT_TEMPLATE = DEFAULT_VISION_PROMPT_TEMPLATE
    elif "vision_prompt_template" in kwargs and kwargs["vision_prompt_template"] is not None:
        vt = str(kwargs["vision_prompt_template"]).strip()
        if vt:
            VISION_PROMPT_TEMPLATE = vt
        else:
            VISION_PROMPT_TEMPLATE = DEFAULT_VISION_PROMPT_TEMPLATE
    elif "custom_prompt" in kwargs and kwargs["custom_prompt"] is not None:
        cp = str(kwargs["custom_prompt"]).strip()
        if cp:
            VISION_PROMPT_TEMPLATE = cp
        else:
            VISION_PROMPT_TEMPLATE = DEFAULT_VISION_PROMPT_TEMPLATE



# Load settings from settings.json or fall back
settings = load_persistent_settings()

saved_inputs = settings.get("INPUT_FOLDERS")
if saved_inputs:
    if isinstance(saved_inputs, str):
        INPUT_FOLDERS = [Path(p.strip()) for p in saved_inputs.split(",") if p.strip()]
    else:
        INPUT_FOLDERS = [Path(str(p).strip()) for p in saved_inputs if str(p).strip()]
else:
    # --- Path Settings ---
    # Root input folders to scan (supports local paths and network UNC paths)
    INPUT_FOLDERS = [
        Path(p.strip()) for p in os.environ.get("INPUT_FOLDERS", "").split(",") if p.strip()
    ]

# If INPUT_FOLDERS is empty, default to scanning 'media_input' in project root
if not INPUT_FOLDERS:
    INPUT_FOLDERS = [PROJECT_ROOT / "media_input"]

if is_running_in_docker():
    cleaned_inputs = []
    for p in INPUT_FOLDERS:
        norm = normalize_container_path(p, fallback_type="input")
        if norm not in cleaned_inputs:
            cleaned_inputs.append(norm)
    INPUT_FOLDERS = cleaned_inputs if cleaned_inputs else [Path("/app/media_input")]

saved_output = settings.get("OUTPUT_FOLDER")
if saved_output and str(saved_output).strip():
    OUTPUT_FOLDER = Path(str(saved_output).strip())
else:
    # Output directory for saving metadata sidecars and database
    OUTPUT_FOLDER = Path(os.environ.get("OUTPUT_FOLDER", str(PROJECT_ROOT / "media_output")))

if is_running_in_docker():
    OUTPUT_FOLDER = normalize_container_path(OUTPUT_FOLDER, fallback_type="output")

# Dedicated folder for saving cropped face images (unrecognized or reference crops)
FACES_FOLDER = OUTPUT_FOLDER / "facess"

# Base storage directory for multi-tenant users
USERS_BASE_DIR: Path = Path(os.environ.get("USERS_BASE_DIR", "/app/storage/users"))

# Whether to preserve relative directory structure in OUTPUT_FOLDER
PRESERVE_STRUCTURE: bool = os.environ.get("PRESERVE_STRUCTURE", "true").lower() in ("true", "1", "yes")

# --- Database Backend Settings ---
# "sqlite" (Option A: default, zero-dependency, FTS5 + vector search) or "postgres"
DB_BACKEND: str = os.environ.get("DB_BACKEND", "sqlite").lower()

# SQLite & Remote / Network Connector Settings
SQLITE_BUSY_TIMEOUT_MS: int = int(os.environ.get("SQLITE_BUSY_TIMEOUT_MS", "60000"))
SQLITE_WAL_MODE: bool = os.environ.get("SQLITE_WAL_MODE", "true").lower() in ("true", "1", "yes")

# Path to SQLite database (supports local files, network mounts, and UNC shares e.g. \\nas\photos\catalog.db)
raw_db_path = os.environ.get("DB_PATH", str(OUTPUT_FOLDER / "catalog_history.db"))
if is_running_in_docker():
    db_s = str(raw_db_path).strip()
    if ":" in db_s or db_s.startswith(("\\\\", "//")):
        DB_PATH = OUTPUT_FOLDER / "catalog_history.db"
    elif not Path(db_s).is_absolute():
        DB_PATH = OUTPUT_FOLDER / Path(db_s).name
    else:
        DB_PATH = Path(db_s)
else:
    DB_PATH = Path(raw_db_path)

# Optional remote database URI (e.g. libsql:// or postgresql://)
DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

# PostgreSQL optional settings
POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT: int = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB: str = os.environ.get("POSTGRES_DB", "media_cataloger")
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", "")
POSTGRES_SSLMODE: str = os.environ.get("POSTGRES_SSLMODE", "prefer")
POSTGRES_POOL_MIN: int = int(os.environ.get("POSTGRES_POOL_MIN", "1"))
POSTGRES_POOL_MAX: int = int(os.environ.get("POSTGRES_POOL_MAX", "10"))
POSTGRES_TIMEOUT: float = float(os.environ.get("POSTGRES_TIMEOUT", "10.0"))

# --- Utility Settings ---
# Executable paths for ffmpeg / ffprobe
FFMPEG_PATH: str = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH: str = os.environ.get("FFPROBE_PATH", "ffprobe")

# Maximum image preview size for Gemini API (along the larger dimension)
IMAGE_MAX_SIZE: int = int(os.environ.get("IMAGE_MAX_SIZE", "1500"))

# Cosine similarity threshold for face recognition (InsightFace)
# Default: 0.70 (70% match threshold)
FACE_SIMILARITY_THRESHOLD: float = float(os.environ.get("FACE_SIMILARITY_THRESHOLD", "0.70"))

# Confidence threshold for face detection / recognition
# Face is saved to facess/ if unrecognized or confidence < 0.70
FACE_CONFIDENCE_THRESHOLD: float = float(os.environ.get("FACE_CONFIDENCE_THRESHOLD", "0.70"))

# --- Gemini API Limits ---
# API Key
raw_gemini_key = settings.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEY: str = decrypt_secret(raw_gemini_key)

# Gemini model name (default: gemini-3.6-flash)
GEMINI_MODEL: str = str(settings.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")).strip()

# Rate limits for Gemini API
RPM_LIMIT: int = int(settings.get("RPM_LIMIT") or os.environ.get("RPM_LIMIT", "15"))  # Requests per minute
TPM_LIMIT: int = int(settings.get("TPM_LIMIT") or os.environ.get("TPM_LIMIT", "1000000"))  # Tokens per minute (increased for video)

# Parallel worker concurrency limits
saved_gemini_workers = settings.get("GEMINI_MAX_WORKERS")
if saved_gemini_workers is not None:
    GEMINI_MAX_WORKERS: int = max(1, int(saved_gemini_workers))
else:
    GEMINI_MAX_WORKERS: int = int(os.environ.get("GEMINI_MAX_WORKERS", "3"))

saved_local_workers = settings.get("LOCAL_MAX_WORKERS")
if saved_local_workers is not None:
    LOCAL_MAX_WORKERS: int = max(1, int(saved_local_workers))
else:
    LOCAL_MAX_WORKERS: int = int(os.environ.get("LOCAL_MAX_WORKERS", "2"))

def get_max_workers(provider: Optional[str] = None) -> int:
    """Return max concurrent workers for the active or given provider."""
    prov = (provider or MODEL_PROVIDER or "gemini").strip().lower()
    if prov == "local":
        return max(1, LOCAL_MAX_WORKERS)
    return max(1, GEMINI_MAX_WORKERS)

# --- Model Provider / Hybrid Settings ---
# Primary provider: "gemini" or "local"
saved_provider = settings.get("MODEL_PROVIDER")
if saved_provider:
    MODEL_PROVIDER: str = str(saved_provider).strip().lower()
else:
    MODEL_PROVIDER: str = os.environ.get("MODEL_PROVIDER", "gemini").lower()

# LM Studio parameters
LM_HOST: str = os.environ.get("LM_HOST", "")
LM_PORT: str = os.environ.get("LM_PORT", "")
LM_API_TOKEN: str = os.environ.get("LM_API_TOKEN", "") or os.environ.get("LOCAL_API_TOKEN", "")
LOCAL_API_TOKEN: str = LM_API_TOKEN

# API base URL for local model (LM Studio)
if LM_HOST:
    host = LM_HOST.rstrip("/")
    if LM_PORT and not (host.endswith(f":{LM_PORT}") or f":{LM_PORT}/" in host):
        LOCAL_API_BASE: str = os.environ.get("LOCAL_API_BASE", f"{host}:{LM_PORT}/v1")
    else:
        LOCAL_API_BASE: str = os.environ.get("LOCAL_API_BASE", host if host.endswith("/v1") else f"{host}/v1")
else:
    LOCAL_API_BASE: str = os.environ.get("LOCAL_API_BASE", "http://localhost:1234/v1")

# Model identifier for local model. If empty, will auto-detect from LM Studio.
saved_local_model = settings.get("LOCAL_MODEL_NAME")
if saved_local_model is not None and str(saved_local_model).strip():
    LOCAL_MODEL_NAME: str = str(saved_local_model).strip()
else:
    LOCAL_MODEL_NAME: str = os.environ.get("LOCAL_MODEL_NAME", "")

# Whether to switch to local model if Gemini quota is reached
FALLBACK_TO_LOCAL: bool = os.environ.get("FALLBACK_TO_LOCAL", "true").lower() in ("true", "1", "yes")

# Model identifier for local Whisper transcription (e.g. large-v3-turbo, medium, base)
saved_whisper = settings.get("WHISPER_MODEL")
if saved_whisper:
    WHISPER_MODEL: str = str(saved_whisper).strip()
else:
    WHISPER_MODEL: str = os.environ.get("WHISPER_MODEL", "large-v3-turbo")

# Device for local Whisper transcription (cuda / cpu)
WHISPER_DEVICE: str = os.environ.get("WHISPER_DEVICE", "cuda")
# UI Base URL for streaming fallback
saved_ui_base = settings.get("UI_BASE_URL")
if saved_ui_base:
    UI_BASE_URL: str = str(saved_ui_base).strip()
else:
    UI_BASE_URL: str = os.environ.get("UI_BASE_URL", "http://localhost:8000")

# AI Engine Tag Recognition and Formatting Configuration
TARGET_TAGS: Optional[List[str]] = None
TAG_FORMAT: str = os.environ.get("TAG_FORMAT", "categorized")

# Unified Vision Prompt Template with dynamic placeholders ({media_type}, {context}, {people}, {tag_instructions})
DEFAULT_VISION_PROMPT_TEMPLATE: str = (
    "Perform a detailed semantic analysis of the {media_type}.\n"
    "Determine the environment type (indoor/outdoor/unknown), lighting characteristics, weather (if outdoor), and time of day.\n"
    "Perform OCR text recognition on any signs or visible text if present.\n"
    "Classify high-level content_type (documents, social, nature, animals, screenshots, family, other) and generate rich semantic tags matching the requested tag format.\n"
    "Fill all main schema fields in English, and provide full Russian translations in the corresponding *_ru fields "
    "so that the output JSON supports searching in both English and Russian.\n\n"
    "{context}\n\n"
    "{people}\n\n"
    "{tag_instructions}"
)

DEFAULT_PHOTO_PROMPT: str = (
    "Perform a detailed semantic analysis of the photo. "
    "Determine the environment type (indoor/outdoor/unknown), lighting characteristics, weather (if outdoor), and time of day. "
    "Perform OCR text recognition on any signs or text if present. "
    "Analyze the provided EXIF metadata (camera model, ISO, shutter speed, aperture, capture time, GPS position) "
    "and provide an expert conclusion in exif_analysis. "
    "Classify the image content_type and generate rich semantic tags matching the requested tag format. "
    "Fill all main schema fields in English, and provide full Russian translations in the corresponding *_ru fields "
    "so that the output JSON supports searching in both English and Russian."
)

DEFAULT_VIDEO_PROMPT: str = (
    "Perform a detailed analysis of the video file plot. "
    "Provide a detailed transcription of speech and key background sounds. "
    "Break the video into logical segments (timeline_events) with exact timecodes (MM:SS format) "
    "and concise activity descriptions. "
    "Classify the video content_type and generate rich semantic tags matching the requested tag format. "
    "Fill all main schema fields in English, and provide full Russian translations in the corresponding *_ru fields "
    "so that the output JSON supports searching in both English and Russian."
)

saved_vision_prompt = settings.get("VISION_PROMPT_TEMPLATE") or settings.get("vision_prompt_template")
if saved_vision_prompt and str(saved_vision_prompt).strip():
    VISION_PROMPT_TEMPLATE: str = str(saved_vision_prompt).strip()
else:
    VISION_PROMPT_TEMPLATE: str = os.environ.get("VISION_PROMPT_TEMPLATE", DEFAULT_VISION_PROMPT_TEMPLATE)

# --- Security, Authentication & Secret Vault Settings ---
CATALOGER_API_URL: str = os.environ.get("CATALOGER_API_URL", "http://localhost:8001").rstrip("/")
JWT_SECRET: str = os.environ.get("JWT_SECRET", "media_cataloger_secure_jwt_secret_key_2026")
AI_SERVICE_USER: str = os.environ.get("AI_SERVICE_USER", "admin")
AI_SERVICE_PASSWORD: str = os.environ.get("AI_SERVICE_PASSWORD", "admin")
VAULT_MASTER_PIN: str = os.environ.get("VAULT_MASTER_PIN", "1234")
VAULT_SESSION_TIMEOUT_SEC: int = int(os.environ.get("VAULT_SESSION_TIMEOUT_SEC", "1800"))  # 30 min default


# Supported file extensions

SUPPORTED_PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Ensure required directories exist if accessible
try:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    FACES_FOLDER.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass



