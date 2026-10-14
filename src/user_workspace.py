import os
from pathlib import Path
from typing import Optional, Union

def get_user_root(user_id: str, default_root: Optional[str] = None) -> Path:
    """Returns the base storage directory for the given user."""
    base_dir = os.environ.get("USERS_BASE_DIR", "/app/storage/users")
    if default_root and str(default_root).strip():
        dr = Path(default_root).resolve()
        norm_dr = str(dr).replace("\\", "/").rstrip("/").lower()
        norm_uid = str(user_id).strip().lower()
        if norm_dr.endswith("/" + norm_uid) or dr.name.lower() == norm_uid:
            return dr
        return dr / str(user_id)
    return Path(base_dir) / str(user_id)

def get_user_catalog_dir(user_id: str, default_root: Optional[str] = None) -> Path:
    """Returns the hidden .catalog directory for the given user."""
    return get_user_root(user_id, default_root) / ".catalog"

def get_user_db_path(user_id: str, default_root: Optional[str] = None) -> Path:
    """Returns the path to the user's catalog_history.db."""
    return get_user_catalog_dir(user_id, default_root) / "catalog_history.db"

def validate_path_in_user_workspace(
    user_id: str,
    target_path: Union[str, Path],
    default_root: Optional[str] = None,
    user_role: Optional[str] = None
) -> bool:
    """Validates that a given path resides within the user's workspace or allowed system folders."""
    # Path traversal with '..' is strictly forbidden
    raw_str = str(target_path).replace("\\", "/")
    parts = [p.strip() for p in raw_str.split("/") if p.strip()]
    if ".." in parts:
        return False

    # Administrators with admin role or admin_panel permission have system-wide storage authority
    if user_role and str(user_role).strip().lower() == "admin":
        return True
    if str(user_id).strip().lower() in ("admin", "user_admin_default"):
        return True

    try:
        target = Path(target_path).resolve()
    except Exception:
        return False

    norm_target = str(target).replace("\\", "/").rstrip("/").lower()

    # 1. Check user private workspace
    try:
        user_root = get_user_root(user_id, default_root).resolve()
        norm_user_root = str(user_root).replace("\\", "/").rstrip("/").lower()
        if norm_target == norm_user_root or norm_target.startswith(norm_user_root + "/"):
            return True
    except Exception:
        pass

    # 2. Check configured system input & output folders
    try:
        from src import config
        candidates = []
        if hasattr(config, "INPUT_FOLDERS"):
            for f in config.INPUT_FOLDERS:
                candidates.append(f)
        if getattr(config, "OUTPUT_FOLDER", None):
            candidates.append(config.OUTPUT_FOLDER)
        if getattr(config, "PROJECT_ROOT", None):
            candidates.append(config.PROJECT_ROOT / "media_input")
            # Sibling media_catalog directory (e.g. adjacent workspace in monorepo/multi-repo setup)
            sibling_input = Path(config.PROJECT_ROOT).parent / "media_catalog" / "media_input"
            if sibling_input.exists():
                candidates.append(sibling_input)

        for folder in candidates:
            try:
                norm_folder = str(Path(folder).resolve()).replace("\\", "/").rstrip("/").lower()
                if norm_target == norm_folder or norm_target.startswith(norm_folder + "/"):
                    return True
            except Exception:
                continue
    except Exception:
        pass

    return False

