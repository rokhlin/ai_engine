import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple, Optional

# Windows CUDA DLL search path helper for nvidia packages
if sys.platform == "win32":
    possible_site_packages = [Path(p) for p in sys.path if "site-packages" in p]
    # Check global python path as fallback
    possible_site_packages.append(Path("C:/Python312/Lib/site-packages"))
    for p in possible_site_packages:
        if p.is_dir():
            nvidia_dirs = [
                p / "nvidia" / "cublas" / "bin",
                p / "nvidia" / "cudnn" / "bin",
                p / "nvidia" / "cuda_runtime" / "bin",
            ]
            for nd in nvidia_dirs:
                if nd.is_dir():
                    try:
                        # Add to DLL directory search path (for Python 3.8+ DLL loads)
                        os.add_dll_directory(str(nd))
                        # Append to PATH environment variable (for ctypes.util.find_library)
                        os.environ["PATH"] = str(nd) + os.pathsep + os.environ["PATH"]
                        print(f"Added DLL directory to search path and PATH: {nd}")
                    except Exception:
                        pass


from faster_whisper import WhisperModel
from src import config

_whisper_model = None

def get_whisper_model() -> WhisperModel:
    """Retrieve or load the local Whisper model on the configured device (cuda/cpu) with fallback."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    model_name = config.WHISPER_MODEL or "large-v3-turbo"
    device = config.WHISPER_DEVICE or "cuda"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"Initializing local Whisper model '{model_name}' on '{device}' (compute_type={compute_type})...")
    
    try:
        _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("Whisper model successfully loaded.")
        return _whisper_model
    except Exception as e:
        print(f"Warning: Failed to load Whisper model on '{device}' with error: {e}")
        if device == "cuda":
            print("Attempting auto-fallback to device='cpu' with compute_type='int8'...")
            try:
                _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
                print("Whisper model successfully loaded on CPU.")
                return _whisper_model
            except Exception as cpu_err:
                print(f"Critical: Failed to load Whisper model on CPU: {cpu_err}")
                raise cpu_err
        else:
            raise e

def translate_text_local(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text from one language to another using the local LLM (LM Studio)."""
    if not text or not text.strip():
        return ""
        
    from src.utils.local_model import call_local_model_api
    from pydantic import BaseModel, Field
    
    class TranslationResult(BaseModel):
        translated_text: str = Field(description="The translated text exactly")
        
    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        "Return ONLY the translation as the specified JSON field. Do not include any explanations, code block ticks, or introduction text."
    )
    
    messages = [
        {"role": "system", "content": "You are a professional multilingual translator. You output exact translations in JSON."},
        {"role": "user", "content": f"{prompt}\n\nText to translate:\n{text}"}
    ]
    
    try:
        res = call_local_model_api(messages, TranslationResult)
        return res.translated_text.strip()
    except Exception as e:
        print(f"Warning: Local translation from {source_lang} to {target_lang} failed: {e}. Returning original text.")
        return text

def transcribe_audio_track(audio_path: Path) -> Tuple[str, str]:
    """Transcribe audio file using Whisper. Returns (transcription_en, transcription_ru)."""
    if not audio_path.is_file():
        return "", ""
        
    global _whisper_model
    model = get_whisper_model()
    print(f"Running speech recognition on {audio_path.name}...")
    
    try:
        # 1. Transcribe (native language)
        segments, info = model.transcribe(str(audio_path), beam_size=5, task="transcribe")
        segments_list = list(segments)  # Trigger lazy loading of execution libraries
    except Exception as e:
        err_str = str(e).lower()
        is_cuda_lib_err = (
            "cublas" in err_str or 
            "cudnn" in err_str or 
            "cuda" in err_str or 
            "cannot be loaded" in err_str or
            "not found" in err_str
        )
        if is_cuda_lib_err and config.WHISPER_DEVICE == "cuda":
            if config.WHISPER_FALLBACK_TO_CPU:
                print(f"\n[WARNING] Whisper CUDA libraries (cuBLAS/cuDNN) are missing or failed to load ({e}).")
                print("Automatically falling back to CPU execution for transcription...")
                config.WHISPER_DEVICE = "cpu"
                _whisper_model = None
                return transcribe_audio_track(audio_path)
            else:
                print(f"\n[FATAL ERROR] Whisper CUDA execution failed and CPU fallback is disabled.\nDetails: {e}")
                import sys
                sys.exit(1)
        else:
            raise e
            
    native_text = " ".join([seg.text for seg in segments_list]).strip()
    
    detected_lang = info.language
    detected_prob = info.language_probability
    print(f"Speech detected: language='{detected_lang}' (probability={detected_prob:.2f}), duration={info.duration:.2f}s")
    
    if not native_text:
        print("No speech detected in audio track.")
        return "", ""
        
    # 2. Derive English and Russian versions
    if detected_lang == "en":
        transcription_en = native_text
        print("Translating transcript to Russian using local model...")
        transcription_ru = translate_text_local(native_text, source_lang="English", target_lang="Russian")
    elif detected_lang == "ru":
        transcription_ru = native_text
        print("Translating transcript to English using Whisper translation task...")
        # Whisper can translate to English natively during transcription
        trans_segments, _ = model.transcribe(str(audio_path), beam_size=5, task="translate")
        transcription_en = " ".join([seg.text for seg in trans_segments]).strip()
    else:
        # Other language: translate to English via Whisper, then English to Russian via LLM
        print(f"Translating native {detected_lang} transcript to English using Whisper...")
        trans_segments, _ = model.transcribe(str(audio_path), beam_size=5, task="translate")
        transcription_en = " ".join([seg.text for seg in trans_segments]).strip()
        
        print("Translating English transcript to Russian using local model...")
        transcription_ru = translate_text_local(transcription_en, source_lang="English", target_lang="Russian")
        
    return transcription_en, transcription_ru

def transcribe_video_audio(video_path: Path) -> Tuple[str, str]:
    """Extract audio from video file and transcribe it. Returns (transcription, transcription_ru)."""
    from src.utils.video import extract_audio_from_video
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "audio.wav"
        try:
            extract_audio_from_video(video_path, audio_path)
            if audio_path.is_file() and audio_path.stat().st_size > 100:
                return transcribe_audio_track(audio_path)
        except Exception as e:
            print(f"Warning: Audio transcription skipped for {video_path.name}: {e}")
            
    return "", ""
