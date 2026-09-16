from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import threading
import os

# Import shared state from api
from api import pipeline_state, state_lock, config

router = APIRouter()

@router.post("/api/select-folder", summary="Open native folder picker")
def select_folder():
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot select folder while synchronization is running.")
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        # Ensure the dialog appears on top
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory(parent=root, title="Select Catalog Folder")
        root.destroy()
        if folder_path:
            return {"folder": str(folder_path)}
        return {"folder": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open folder picker: {str(e)}")

@router.post("/api/select-file", summary="Open native file picker")
def select_file():
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Cannot select file while synchronization is running.")
    try:
        import tkinter
        from tkinter import filedialog
        root = tkinter.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Select Media File",
            filetypes=[
                ("Media files", "*.jpg;*.jpeg;*.png;*.webp;*.heic;*.mp4;*.mov;*.avi;*.mkv"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        if file_path:
            return {"file": str(file_path)}
        return {"file": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open file picker: {str(e)}")
