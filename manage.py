#!/usr/bin/env python3
"""
Media Cataloger (AI Engine) - Task Runner & Management Script
Provides quick commands for running, building, testing, and developing the AI Engine and database.

Usage:
    python manage.py [command] [--env <env_file>] [options...]

Examples:
    python manage.py api                     # Run FastAPI remote control daemon locally with auto-reload
    python manage.py scan --force            # Run cataloger scanner locally
    python manage.py test                    # Run pytest test suite
    python manage.py db:status               # View database migrations and stats
    python manage.py db:backup               # Hot-backup SQLite database
    python manage.py db:migrate              # Apply pending migrations
    python manage.py up                      # Start Docker container in background
    python manage.py down                    # Stop Docker container
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Color styling for CLI output
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def log_info(msg: str):
    print(f"{Colors.CYAN}[INFO]{Colors.RESET} {msg}")

def log_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.RESET} {msg}")

def log_warn(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.RESET} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {msg}", file=sys.stderr)

PROJECT_ROOT = Path(__file__).resolve().parent

def resolve_env_file(env_arg: str = None) -> Path:
    """Resolve which environment configuration file to use."""
    if env_arg:
        env_path = Path(env_arg)
        if not env_path.is_absolute():
            env_path = PROJECT_ROOT / env_path
        if not env_path.is_file():
            log_warn(f"Specified environment file '{env_arg}' not found. Will proceed if system env is set.")
        return env_path

    # Automatic detection order (data/config/.env, .env.local, .env)
    config_dir = PROJECT_ROOT / "data" / "config"
    if (config_dir / ".env.local").is_file():
        return config_dir / ".env.local"
    if (config_dir / ".env").is_file():
        return config_dir / ".env"
    if (PROJECT_ROOT / ".env.local").is_file():
        return PROJECT_ROOT / ".env.local"
    if (PROJECT_ROOT / ".env").is_file():
        return PROJECT_ROOT / ".env"
    
    return config_dir / ".env"

def load_env_vars(env_path: Path) -> dict:
    """Read key-values from env file without external dependencies."""
    env_vars = {}
    if env_path.is_file():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip().strip('"').strip("'")
    return env_vars

def get_python_exec() -> str:
    """Get the active Python executable or venv Python."""
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        return str(venv_py)
    venv_py_nix = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_py_nix.is_file():
        return str(venv_py_nix)
    return sys.executable

def run_cmd(cmd: list, env_file: Path = None, extra_env: dict = None) -> int:
    """Run a subprocess command with appropriate environment variables."""
    full_env = os.environ.copy()
    
    if env_file and env_file.is_file():
        full_env["ENV_FILE"] = str(env_file)
        loaded = load_env_vars(env_file)
        full_env.update(loaded)
    
    if extra_env:
        full_env.update(extra_env)
        
    cmd_str = " ".join(cmd)
    log_info(f"Executing: {Colors.BOLD}{cmd_str}{Colors.RESET}")
    try:
        res = subprocess.run(cmd, env=full_env, cwd=PROJECT_ROOT)
        return res.returncode
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return 130
    except FileNotFoundError as e:
        log_error(f"Command not found: {cmd[0]}. Please ensure it is installed and in your PATH. ({e})")
        return 127

# =====================================================================
# Command Implementations
# =====================================================================

def cmd_up(env_file: Path, extra_args: list):
    """Run Cataloger AI Docker container in background."""
    log_info(f"Starting Cataloger AI Docker container using config: {Colors.BOLD}{env_file.name}{Colors.RESET}")
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    
    cmd = [
        "docker", "compose",
        "-f", str(compose_file),
        "--env-file", str(env_file),
        "up", "-d"
    ]
    if "--build" not in extra_args:
        cmd.append("--build")
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_up_cataloger(env_file: Path, extra_args: list):
    """Start Media Cataloger AI Service container (Machine B)."""
    return cmd_up(env_file, extra_args)

def cmd_up_frontend(env_file: Path, extra_args: list):
    """Start Media Library Web UI note."""
    log_info("Media Library Web UI is now hosted in standalone repository: https://github.com/rokhlin/media_cataloger_web")
    return 0

def cmd_up_all(env_file: Path, extra_args: list):
    """Start all containers on single host."""
    log_info(f"Starting all containers using config: {Colors.BOLD}{env_file.name}{Colors.RESET}")
    compose_file = PROJECT_ROOT / "docker-compose.all.yml"
    cmd = ["docker", "compose", "-f", str(compose_file), "--env-file", str(env_file), "up", "-d"]
    if "--build" not in extra_args:
        cmd.append("--build")
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_down(env_file: Path, extra_args: list):
    """Stop running Docker containers."""
    log_info("Stopping Docker containers...")
    cmd = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml")]
    if (PROJECT_ROOT / "docker-compose.all.yml").is_file():
        cmd.extend(["-f", str(PROJECT_ROOT / "docker-compose.all.yml")])
    cmd.append("down")
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_build(env_file: Path, extra_args: list):
    """Build Media Cataloger AI Docker image."""
    log_info("Building Media Cataloger AI Docker image...")
    cmd = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), "build"]
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_build_cataloger(env_file: Path, extra_args: list):
    """Build Cataloger backend Docker image."""
    return cmd_build(env_file, extra_args)

def cmd_build_frontend(env_file: Path, extra_args: list):
    """Build Frontend image note."""
    log_info("Frontend is now hosted in standalone repository https://github.com/rokhlin/media_cataloger_web")
    return 0

def cmd_logs(env_file: Path, extra_args: list):
    """Follow Docker logs."""
    cmd = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), "logs", "-f"]
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_api(env_file: Path, extra_args: list):
    """Run Python FastAPI remote control daemon locally on port 8001."""
    log_info(f"Running Media Cataloger API server using config: {Colors.BOLD}{env_file.name}{Colors.RESET}")
    py = get_python_exec()
    env_vars = load_env_vars(env_file)
    port = env_vars.get("API_PORT", "8001")
    cmd = [py, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", port, "--reload"]
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_frontend(env_file: Path, extra_args: list):
    """Forward frontend command note."""
    log_info("Media Library Web UI has moved to https://github.com/rokhlin/media_cataloger_web")
    return 0

def cmd_scan(env_file: Path, extra_args: list):
    """Run cataloging pipeline CLI locally."""
    log_info(f"Running media scan using config: {Colors.BOLD}{env_file.name}{Colors.RESET}")
    py = get_python_exec()
    cmd = [py, "main.py", "run"]
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_test(env_file: Path, extra_args: list):
    """Run test suite."""
    log_info("Running test suite (pytest)...")
    py = get_python_exec()
    cmd = [py, "-m", "pytest"]
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_verify(env_file: Path, extra_args: list):
    """Run system verification script."""
    log_info("Running verification script...")
    py = get_python_exec()
    cmd = [py, "verify_run.py"]
    cmd.extend(extra_args)
    return run_cmd(cmd, env_file=env_file)

def cmd_info(env_file: Path, extra_args: list):
    """Print current configuration and resolved variables."""
    print(f"\n{Colors.BOLD}=== Media Cataloger Environment Info ==={Colors.RESET}")
    print(f"Project Root     : {PROJECT_ROOT}")
    print(f"Active Env File  : {Colors.GREEN}{env_file}{Colors.RESET} ({'EXISTS' if env_file.is_file() else 'MISSING'})")
    print(f"Python Executable: {get_python_exec()}")
    print("-" * 50)
    
    env_vars = load_env_vars(env_file)
    if not env_vars:
        print(f"{Colors.YELLOW}No variables loaded from {env_file.name}{Colors.RESET}")
    else:
        for k, v in sorted(env_vars.items()):
            if "KEY" in k or "SECRET" in k or "PASSWORD" in k:
                display_v = f"{v[:6]}...{v[-4:]}" if len(v) > 12 else "******"
            else:
                display_v = v
            print(f"{k:<25} = {display_v}")
    print("-" * 50)
    return 0

def get_db_path(env_file: Path) -> Path:
    """Resolve database path from environment configuration or default."""
    env_vars = load_env_vars(env_file)
    db_path_str = env_vars.get("DB_PATH")
    if not db_path_str:
        out_folder = env_vars.get("OUTPUT_FOLDER", "./media_output")
        db_path_str = f"{out_folder}/catalog_history.db"
    
    db_path = Path(db_path_str)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return db_path

def cmd_db_status(env_file: Path, extra_args: list):
    """Display database health, applied migrations, pending migrations, and table stats."""
    from src.migrations import get_migration_status
    import sqlite3
    
    db_path = get_db_path(env_file)
    print(f"\n{Colors.BOLD}=== Media Cataloger Database Status ==={Colors.RESET}")
    print(f"Target DB Path  : {Colors.CYAN}{db_path}{Colors.RESET}")
    
    if not db_path.is_file():
        print(f"Status          : {Colors.YELLOW}Database file does not exist yet (will be initialized on first run){Colors.RESET}")
        return 0
        
    size_mb = round(db_path.stat().st_size / (1024 * 1024), 2)
    print(f"File Size       : {size_mb} MB")
    
    status = get_migration_status(db_path)
    print(f"Migration Status: {status['applied_count']} applied, {status['pending_count']} pending (Latest Version: v{status.get('latest_version', 0)})")
    
    if status["applied"]:
        print(f"\n{Colors.BOLD}Applied Migrations:{Colors.RESET}")
        for m in status["applied"]:
            print(f"  {Colors.GREEN}[OK]{Colors.RESET} [{m['version']:03d}] {m['name']} (applied: {m['applied_at']}, {m.get('duration_ms', 0)}ms)")
            
    if status["pending"]:
        print(f"\n{Colors.BOLD}Pending Migrations:{Colors.RESET}")
        for m in status["pending"]:
            print(f"  {Colors.YELLOW}[PENDING]{Colors.RESET} [{m['version']:03d}] {m['name']} (run 'python manage.py db:migrate' to apply)")
            
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        cur = conn.cursor()
        print(f"\n{Colors.BOLD}Table Statistics:{Colors.RESET}")
        tables = ["media_items", "media_metadata", "persons", "face_registry", "media_faces", "media_tags", "video_timeline_events", "sync_history"]
        for tbl in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tbl};")
                count = cur.fetchone()[0]
                print(f"  - {tbl:<22}: {count:,} rows")
            except Exception:
                pass
        conn.close()
    except Exception as e:
        log_warn(f"Could not read table statistics: {e}")
        
    print("-" * 50)
    return 0

def cmd_db_backup(env_file: Path, extra_args: list):
    """Create a safe, atomic hot backup of the SQLite database."""
    from src.migrations import backup_database
    db_path = get_db_path(env_file)
    
    if not db_path.is_file():
        log_error(f"Cannot backup: database file does not exist at {db_path}")
        return 1
        
    log_info(f"Creating hot backup of database: {db_path}...")
    backup_file = backup_database(db_path)
    if backup_file:
        log_success(f"Backup created successfully: {Colors.BOLD}{backup_file}{Colors.RESET}")
        return 0
    else:
        log_error("Database backup failed.")
        return 1

def cmd_db_migrate(env_file: Path, extra_args: list):
    """Apply all pending database schema migrations with automated pre-backup."""
    from src.migrations import run_migrations
    db_path = get_db_path(env_file)
    
    log_info(f"Applying pending migrations for: {Colors.CYAN}{db_path}{Colors.RESET}...")
    result = run_migrations(db_path, auto_backup=True)
    if result.get("applied_count", 0) >= 0:
        log_success(f"Database schema updated successfully ({result.get('applied_count', 0)} migrations applied).")
        return 0
    else:
        log_error("Migration failed. Please inspect error logs above.")
        return 1

# Command mapping table
COMMANDS = {
    "api": cmd_api,
    "cataloger": cmd_api,
    "backend": cmd_api,
    "frontend": cmd_frontend,
    "scan": cmd_scan,
    "test": cmd_test,
    "verify": cmd_verify,
    "info": cmd_info,
    "up": cmd_up,
    "up-cataloger": cmd_up_cataloger,
    "up-frontend": cmd_up_frontend,
    "up-all": cmd_up_all,
    "down": cmd_down,
    "build": cmd_build,
    "build-cataloger": cmd_build_cataloger,
    "build-frontend": cmd_build_frontend,
    "logs": cmd_logs,
    "db": cmd_db_status,
    "db:status": cmd_db_status,
    "db:backup": cmd_db_backup,
    "db:migrate": cmd_db_migrate,
}

COMMAND_MAP = COMMANDS

def print_help():
    print(f"""
{Colors.BOLD}Media Cataloger (AI Engine) - Task Runner{Colors.RESET}

{Colors.BOLD}Usage:{Colors.RESET}
    python manage.py <command> [--env <env_file>] [options...]

{Colors.BOLD}Available Commands:{Colors.RESET}
    {Colors.CYAN}api{Colors.RESET} / {Colors.CYAN}cataloger{Colors.RESET}   Run FastAPI remote control daemon locally (:8001)
    {Colors.CYAN}scan{Colors.RESET}                Run media cataloging scan locally via CLI
    {Colors.CYAN}test{Colors.RESET}                Run automated test suite (pytest)
    {Colors.CYAN}verify{Colors.RESET}              Run end-to-end verification script
    {Colors.CYAN}db:status{Colors.RESET}           View SQLite database migrations and row counts
    {Colors.CYAN}db:backup{Colors.RESET}           Create atomic hot backup of production database
    {Colors.CYAN}db:migrate{Colors.RESET}          Apply pending database schema migrations
    {Colors.CYAN}up{Colors.RESET}                  Start Cataloger Docker container in background
    {Colors.CYAN}down{Colors.RESET}                Stop Docker containers
    {Colors.CYAN}build{Colors.RESET}               Build Cataloger Docker image locally
    {Colors.CYAN}logs{Colors.RESET}                Follow Docker container logs
    {Colors.CYAN}info{Colors.RESET}                Display resolved paths and active configuration

{Colors.BOLD}Note for Frontend:{Colors.RESET}
    The web UI, Family Tree, and media viewer have moved to the standalone repository:
    {Colors.GREEN}https://github.com/rokhlin/media_cataloger_web{Colors.RESET}
""")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", nargs="?", default="help", help="Command to run")
    parser.add_argument("-e", "--env", "--env-file", dest="env_file", default=None, help="Path to custom .env file")
    
    args, extra_args = parser.parse_known_args()
    
    cmd_name = args.command.lower()
    
    if cmd_name in ["help", "-h", "--help"]:
        print_help()
        sys.exit(0)
        
    if cmd_name not in COMMANDS:
        log_error(f"Unknown command: '{cmd_name}'. Run 'python manage.py help' for available commands.")
        sys.exit(1)
        
    env_path = resolve_env_file(args.env_file)
    handler = COMMANDS[cmd_name]
    ret = handler(env_path, extra_args)
    sys.exit(ret if isinstance(ret, int) else 0)

if __name__ == "__main__":
    main()
