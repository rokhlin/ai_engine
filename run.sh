#!/usr/bin/env bash
# Quick runner script for Media Cataloger on Linux / macOS / WSL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    PYTHON_EXE="python3"
fi

CMD="${1:-dev}"
shift || true
exec "$PYTHON_EXE" "$SCRIPT_DIR/manage.py" "$CMD" "$@"

