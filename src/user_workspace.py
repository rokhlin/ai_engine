import os
from pathlib import Path
from typing import Optional

def get_user_root(user_id: str, default_root: Optional[str] = None) -> Path:
    """Returns the base storage directory for the given user."""
    base_dir = os.environ.get("USERS_BASE_DIR", "/app/storage/users")
    if default_root:
        base_dir = default_root
    return Path(base_dir) / str(user_id)

def get_user_catalog_dir(user_id: str, default_root: Optional[str] = None) -> Path:
    """Returns the hidden .catalog directory for the given user."""
    return get_user_root(user_id, default_root) / ".catalog"

def get_user_db_path(user_id: str, default_root: Optional[str] = None) -> Path:
    """Returns the path to the user's catalog_history.db."""
    return get_user_catalog_dir(user_id, default_root) / "catalog_history.db"

def validate_path_in_user_workspace(user_id: str, target_path: Path, default_root: Optional[str] = None) -> bool:
    """Validates that a given path resides within the user's workspace."""
    user_root = get_user_root(user_id, default_root).resolve()
    try:
        resolved_target = target_path.resolve()
        return str(resolved_target).startswith(str(user_root))
    except Exception:
        return False
