import subprocess
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from src import config

def _run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    """Helper method to run external CLI commands."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command {' '.join(cmd)}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Command failed with error: {e.stderr}") from e

def is_ffmpeg_available() -> bool:
    """Check availability of ffmpeg and ffprobe."""
    try:
        subprocess.run([config.FFMPEG_PATH, "-version"], capture_output=True, check=True)
        subprocess.run([config.FFPROBE_PATH, "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def read_video_metadata(file_path: Path) -> Dict[str, Any]:
    """Read video container metadata via ffprobe."""
    meta = {
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "bitrate": 0,
        "codec": None,
        "datetime": None
    }
    
    cmd = [
        config.FFPROBE_PATH,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,codec_name",
        "-show_entries", "format=duration,size,bit_rate,tags",
        "-of", "json",
        str(file_path)
    ]
    
    try:
        res = _run_cmd(cmd)
        data = json.loads(res.stdout)
        
        # Parse format
        fmt = data.get("format", {})
        meta["duration"] = float(fmt.get("duration", 0.0))
        meta["bitrate"] = int(fmt.get("bit_rate", 0))
        
        # Tags and creation date
        tags = fmt.get("tags", {})
        # Priority creation date tags
        meta["datetime"] = tags.get("creation_time") or tags.get("DateTime")
        
        # Parse video streams
        streams = data.get("streams", [])
        if streams:
            v_stream = streams[0]
            meta["width"] = int(v_stream.get("width", 0))
            meta["height"] = int(v_stream.get("height", 0))
            meta["codec"] = v_stream.get("codec_name")
            
            # Parse FPS from string like "24/1" or "30000/1001"
            fps_str = v_stream.get("avg_frame_rate", "0/0")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                if float(den) > 0:
                    meta["fps"] = float(num) / float(den)
                    
    except Exception as e:
        print(f"Error reading video metadata for {file_path.name}: {e}")
        
    return meta

def compress_video_for_cloud(input_path: Path, output_path: Path) -> Path:
    """Compress video to 720p / 24fps for upload to Gemini File API."""
    if not is_ffmpeg_available():
        raise RuntimeError(
            f"ffmpeg or ffprobe not found! Check that they are installed and specified in config.py.\n"
            f"Current values: FFMPEG_PATH='{config.FFMPEG_PATH}', FFPROBE_PATH='{config.FFPROBE_PATH}'"
        )
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Compress video: max height 720, fps=24, H.264 codec
    cmd = [
        config.FFMPEG_PATH,
        "-y",
        "-i", str(input_path),
        "-vf", "scale=-2:'min(720,ih)'",
        "-r", "24",
        "-c:v", "libx264",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path)
    ]
    
    print(f"Compressing video {input_path.name} -> {output_path.name}...")
    _run_cmd(cmd)
    return output_path

def extract_frames_at_1fps(video_path: Path, output_dir: Path) -> List[Path]:
    """Extract frames from video at 1 frame per second."""
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available for frame extraction.")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract 1 frame per second
    # Filename format: frame_0001.jpg, frame_0002.jpg, etc.
    cmd = [
        config.FFMPEG_PATH,
        "-y",
        "-i", str(video_path),
        "-vf", "fps=1",
        str(output_dir / "frame_%04d.jpg")
    ]
    
    print(f"Extracting 1fps frames for {video_path.name}...")
    _run_cmd(cmd)
    
    # Return sorted list of frame file paths
    frames = sorted(list(output_dir.glob("frame_*.jpg")))
    return frames

def extract_audio_from_video(video_path: Path, output_audio_path: Path) -> Path:
    """Extract mono audio track from video optimized for local transcription."""
    if not is_ffmpeg_available():
        raise RuntimeError("ffmpeg is not available for audio extraction.")
        
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract mono audio, 16kHz sample rate, standard PCM 16-bit WAV
    cmd = [
        config.FFMPEG_PATH,
        "-y",
        "-i", str(video_path),
        "-vn",                   # No video stream
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", "16000",          # 16kHz
        "-ac", "1",              # Mono
        str(output_audio_path)
    ]
    
    print(f"Extracting audio track for {video_path.name}...")
    _run_cmd(cmd)
    return output_audio_path


