import base64
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union
import requests
from pydantic import BaseModel
from src import config

_cached_model_name = None

def get_loaded_local_models() -> List[str]:
    """Retrieve list of model keys/identifiers currently loaded in LM Studio memory."""
    loaded_models: List[str] = []

    # 1. Try lms ps --json
    lms_path = shutil.which("lms")
    if lms_path:
        try:
            res = subprocess.run([lms_path, "ps", "--json"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if isinstance(data, list):
                    for item in data:
                        if item.get("type") == "embedding":
                            continue
                        key = item.get("modelKey") or item.get("identifier") or item.get("id") or item.get("path")
                        if key:
                            loaded_models.append(str(key))
                    if loaded_models:
                        return loaded_models
        except Exception:
            pass

    # 2. Fallback to LM Studio REST API /models
    try:
        url = f"{config.LOCAL_API_BASE.rstrip('/')}/models"
        headers = {}
        token = (getattr(config, "LOCAL_API_TOKEN", "") or getattr(config, "LM_API_TOKEN", "") or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                for m in data["data"]:
                    mid = m.get("id") or m.get("modelKey")
                    if mid:
                        loaded_models.append(str(mid))
    except Exception as e:
        print(f"[Local Model] Failed to query {config.LOCAL_API_BASE}/models: {e}")

    return loaded_models

def get_local_model_name() -> str:
    """Retrieve the vision/chat model currently loaded in LM Studio.
    
    CRITICAL RULE: Never trigger model load or unload during file analysis.
    Uses the loaded model directly, or raises RuntimeError if no model is loaded in LM Studio.
    """
    global _cached_model_name

    loaded = get_loaded_local_models()
    if not loaded:
        _cached_model_name = None
        raise RuntimeError(
            "No model is currently loaded in LM Studio memory. "
            "Please select and load a vision model in Settings before starting analysis."
        )

    # If cached name is still in the currently loaded models, reuse it
    if _cached_model_name and _cached_model_name in loaded:
        return _cached_model_name

    # Priority 1: Configured model if it is currently loaded in LM Studio
    if config.LOCAL_MODEL_NAME and config.LOCAL_MODEL_NAME.strip():
        cfg_target = config.LOCAL_MODEL_NAME.strip().lower()
        for mid in loaded:
            if mid.lower() == cfg_target or cfg_target in mid.lower() or mid.lower() in cfg_target:
                _cached_model_name = mid
                print(f"[Local Model] Using loaded configured model in LM Studio: {_cached_model_name}")
                return _cached_model_name

    # Priority 2: Explicitly vision-capable loaded models
    vision_models = [
        mid for mid in loaded 
        if any(v in mid.lower() for v in ["vl", "vision", "caption", "llava", "multimodal", "gemma-4", "olmocr"])
        and not any(ex in mid.lower() for ex in ["whisper", "embed", "nomic", "tts", "stt"])
    ]
    if vision_models:
        _cached_model_name = vision_models[0]
        print(f"[Local Model] Auto-detected active vision model in LM Studio: {_cached_model_name}")
        return _cached_model_name

    # Priority 3: Any general LLM loaded (excluding audio/embeddings)
    valid_models = [
        mid for mid in loaded 
        if not any(ex in mid.lower() for ex in ["whisper", "embed", "nomic", "tts", "stt"])
    ]
    if valid_models:
        _cached_model_name = valid_models[0]
        print(f"[Local Model] Using active model in LM Studio: {_cached_model_name}")
        return _cached_model_name

    # Fallback to the first loaded model
    _cached_model_name = loaded[0]
    print(f"[Local Model] Using loaded model in LM Studio: {_cached_model_name}")
    return _cached_model_name

def switch_local_model(target_model: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Safely switch the local model in LM Studio.
    
    1. If target_model is already loaded, reuses it immediately without reloading.
    2. If analysis is currently running, waits up to timeout seconds for it to become idle.
    3. Executes 'lms unload -a' to completely free VRAM/RAM.
    4. Executes 'lms load <target_model> -y' to load the new model.
    """
    global _cached_model_name
    if not target_model or not str(target_model).strip():
        return {"success": False, "message": "Target model identifier cannot be empty."}

    clean_target = str(target_model).strip()
    print(f"[Local Model] Model switch requested to: '{clean_target}'")

    # Step 1: Check if already loaded
    loaded = get_loaded_local_models()
    clean_lower = clean_target.lower()
    for mid in loaded:
        if mid.lower() == clean_lower or clean_lower in mid.lower() or mid.lower() in clean_lower:
            _cached_model_name = mid
            config.LOCAL_MODEL_NAME = mid
            msg = f"Model '{mid}' is already loaded in LM Studio memory. Reusing active instance."
            print(f"[Local Model] {msg}")
            return {"success": True, "message": msg, "model": mid}

    # Step 2: Check if analysis is currently active and wait for idle
    try:
        from api import pipeline_state, state_lock
        start_wait = time.time()
        is_busy = False
        with state_lock:
            if pipeline_state.get("status") == "running":
                is_busy = True

        if is_busy:
            print(f"[Local Model] Analysis pipeline currently running. Waiting up to {timeout}s for task to complete before unloading previous model...")
            while time.time() - start_wait < timeout:
                time.sleep(1.0)
                with state_lock:
                    if pipeline_state.get("status") != "running":
                        is_busy = False
                        break
            if is_busy:
                msg = f"Cannot switch model to '{clean_target}': active analysis task did not finish within {timeout}s."
                print(f"[Local Model] [ERROR] {msg}")
                return {"success": False, "message": msg}
    except Exception as check_err:
        print(f"[Local Model] Pipeline status check warning: {check_err}")

    # Step 3: Unload previous models from LM Studio to free GPU VRAM/RAM
    lms_path = shutil.which("lms")
    print("[Local Model] Unloading previous models from LM Studio to free GPU VRAM/RAM...")
    if lms_path:
        try:
            unload_res = subprocess.run([lms_path, "unload", "-a"], capture_output=True, text=True, timeout=30)
            print(f"[Local Model] lms unload output: {unload_res.stdout.strip()}")
        except Exception as unload_err:
            print(f"[Local Model] Warning during lms unload: {unload_err}")
    else:
        try:
            url = f"{config.LOCAL_API_BASE.rstrip('/')}/models/unload"
            headers = {"Content-Type": "application/json"}
            token = (getattr(config, "LOCAL_API_TOKEN", "") or getattr(config, "LM_API_TOKEN", "") or "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            requests.post(url, headers=headers, json={"all": True}, timeout=10)
        except Exception:
            pass

    # Step 4: Load new model into LM Studio
    print(f"[Local Model] Loading model '{clean_target}' into LM Studio...")
    if lms_path:
        try:
            load_res = subprocess.run([lms_path, "load", clean_target, "-y"], capture_output=True, text=True, timeout=60)
            if load_res.returncode == 0:
                _cached_model_name = clean_target
                config.LOCAL_MODEL_NAME = clean_target
                msg = f"Successfully loaded model '{clean_target}' into LM Studio."
                print(f"[Local Model] {msg}")
                return {"success": True, "message": msg, "model": clean_target}
            else:
                err = (load_res.stderr or load_res.stdout or "Unknown error").strip()
                print(f"[Local Model] lms load returned error: {err}")
                return {"success": False, "message": f"Failed to load model '{clean_target}': {err}"}
        except Exception as load_err:
            print(f"[Local Model] Failed to execute lms load: {load_err}")
            return {"success": False, "message": f"Execution error loading model: {load_err}"}
    else:
        try:
            url = f"{config.LOCAL_API_BASE.rstrip('/')}/models/load"
            headers = {"Content-Type": "application/json"}
            token = (getattr(config, "LOCAL_API_TOKEN", "") or getattr(config, "LM_API_TOKEN", "") or "").strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            resp = requests.post(url, headers=headers, json={"model": clean_target}, timeout=30)
            if resp.ok:
                _cached_model_name = clean_target
                config.LOCAL_MODEL_NAME = clean_target
                msg = f"Loaded model '{clean_target}' via LM Studio API."
                print(f"[Local Model] {msg}")
                return {"success": True, "message": msg, "model": clean_target}
            else:
                return {"success": False, "message": f"API load returned status {resp.status_code}: {resp.text}"}
        except Exception as api_err:
            return {"success": False, "message": f"Failed to load model via API: {api_err}"}

def encode_image_to_base64(image_path: Path) -> str:
    """Scale image and convert to base64-encoded string."""
    from src.utils.gemini import prepare_image_bytes
    img_bytes = prepare_image_bytes(image_path)
    return base64.b64encode(img_bytes).decode("utf-8")

def call_local_model_api(
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    temperature: float = 0.1
) -> BaseModel:
    """Call LM Studio API with JSON Schema/JSON Object format and validate/parse the response."""
    model_name = get_local_model_name()
    url = f"{config.LOCAL_API_BASE.rstrip('/')}/chat/completions"
    
    schema = response_model.model_json_schema()
    headers = {
        "Content-Type": "application/json"
    }
    token = (getattr(config, "LOCAL_API_TOKEN", "") or getattr(config, "LM_API_TOKEN", "") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Try with strict json_schema response format first
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": schema
            }
        }
    }
    
    print(f"Sending request to local model API (model: {model_name})...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result_data = response.json()
        content = result_data["choices"][0]["message"]["content"]
        
        print(f"Received response from local model API ({len(content)} chars)")
        
        try:
            parsed_json = json.loads(content)
            return response_model.model_validate(parsed_json)
        except Exception as parse_err:
            print(f"Warning: Failed to parse JSON output directly: {parse_err}")
            cleaned_content = content.strip()
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```", 1)[1].split("```", 1)[0].strip()
            
            parsed_json = json.loads(cleaned_content)
            return response_model.model_validate(parsed_json)
            
    except Exception as api_err:
        if 'response' in locals() and hasattr(response, 'text') and response.text:
            print(f"[ERROR Details from Local Model API]: {response.text}")
        print(f"Local API Call failed with JSON schema. Retrying with direct prompt instructions: {api_err}")
        
        # Prepare fallback payload (standard text mode with prompt schema injection)
        payload_fallback = {
            "model": model_name,
            "messages": [dict(m) for m in messages],
            "temperature": temperature,
            "response_format": {"type": "text"}
        }
        
        # Inject schema instruction into the user prompt
        for msg in reversed(payload_fallback["messages"]):
            if msg["role"] == "user":
                schema_instruction = f"\n\nReturn output ONLY as a JSON object matching this schema:\n{json.dumps(schema, indent=2)}"
                if isinstance(msg["content"], str):
                    msg["content"] += schema_instruction
                elif isinstance(msg["content"], List):
                    msg["content"] = list(msg["content"]) + [{
                        "type": "text",
                        "text": schema_instruction
                    }]
                break
                
        try:
            response = requests.post(url, headers=headers, json=payload_fallback, timeout=300)
            response.raise_for_status()
            result_data = response.json()
            content = result_data["choices"][0]["message"]["content"]
            
            cleaned_content = content.strip()
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```", 1)[1].split("```", 1)[0].strip()
                
            parsed_json = json.loads(cleaned_content)
            return response_model.model_validate(parsed_json)
        except Exception as fallback_err:
            if 'response' in locals() and hasattr(response, 'text') and response.text:
                print(f"[ERROR Details from Local Model API during fallback]: {response.text}")
            raise fallback_err


def analyze_photo_local(
    image_path: Path, 
    exif_data: Optional[Dict[str, Any]] = None,
    target_tags: Optional[List[str]] = None,
    tag_format: Optional[str] = "categorized",
    faces_data: Optional[List[Dict[str, Any]]] = None,
    custom_prompt: Optional[str] = None
) -> Any:
    """Analyze single photo locally via LM Studio."""
    from src.utils.gemini import PhotoAnalysis, format_people_prompt_section, build_vision_prompt
    
    base64_img = encode_image_to_base64(image_path)
    
    exif_prompt_part = ""
    if exif_data:
        exif_prompt_part = (
            "EXIF metadata for this shot:\n"
            f"{json.dumps(exif_data, ensure_ascii=False, indent=2)}"
        )
        
    tag_instructions = ""
    if target_tags:
        target_tags_str = ", ".join(f"'{t}'" for t in target_tags)
        tag_instructions += (
            f"Target tags criteria to evaluate against: [{target_tags_str}]. "
            "If the photo matches any of these target tags/categories, assign them with appropriate confidence scores (0.0 - 1.0). "
        )
    if tag_format == "flat":
        tag_instructions += "\nTag format: output concise keywords in 'tag' field, category 'general'."
    elif tag_format == "prefixed":
        tag_instructions += "\nTag format: populate tag using 'category:tag' naming convention."
    else:
        tag_instructions += (
            "\nTag format: populate 'tag', explicit 'category' "
            "(e.g. content_type, event, objects, scene, people, mood), and 'confidence' (0.0 to 1.0). "
            "Also classify high-level 'content_type' (documents, social, nature, animals, screenshots, family, other)."
        )

    people_str = format_people_prompt_section(faces_data, is_video=False)
    prompt = build_vision_prompt(
        media_type="photo",
        context_str=exif_prompt_part,
        people_str=people_str,
        tag_instructions=tag_instructions,
        custom_template=custom_prompt
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                }
            ]
        }
    ]
    
    return call_local_model_api(messages, PhotoAnalysis)

def analyze_video_local(
    video_path: Path, 
    frames: Optional[List[Path]] = None,
    transcription: Optional[str] = None,
    transcription_ru: Optional[str] = None,
    target_tags: Optional[List[str]] = None,
    tag_format: Optional[str] = "categorized",
    faces_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    custom_prompt: Optional[str] = None
) -> Any:
    """Analyze video based on chronological frame samples and speech transcription via LM Studio."""
    from src.utils.gemini import VideoAnalysis, format_people_prompt_section, build_vision_prompt
    
    # Auto-transcribe audio locally using Whisper if not already provided
    if transcription is None or transcription_ru is None:
        try:
            from src.utils.transcribe import transcribe_video_audio
            transcription, transcription_ru = transcribe_video_audio(video_path)
        except Exception as t_err:
            print(f"Warning: Automatic local audio transcription failed: {t_err}")

    
    if not frames:
        return VideoAnalysis(
            summary="Video metadata parsed. Scene analysis skipped because no frames were extracted.",
            summary_ru="Метаданные видео проанализированы. Анализ сцен пропущен, так как кадры не были извлечены.",
            transcription=transcription or "[Transcription not available: no frames provided]",
            transcription_ru=transcription_ru or "[Транскрипция недоступна: кадры не предоставлены]",
            timeline_events=[],
            content_type="other",
            tags=[]
        )
        
    # Sample up to 4 frames evenly spaced to prevent context window exhaustion and timeouts
    num_frames = len(frames)
    max_samples = min(4, num_frames)
    if max_samples <= 0:
        sampled_indices = []
    elif max_samples == 1:
        sampled_indices = [0]
    else:
        sampled_indices = [int(i * (num_frames - 1) / (max_samples - 1)) for i in range(max_samples)]
        
    sampled_indices = sorted(list(set(sampled_indices)))
    
    transcription_prompt = ""
    if transcription:
        transcription_prompt = (
            "Transcription of spoken audio in this video:\n"
            f"{transcription}\n"
            "Use this transcription to help understand context, dialogue, and plot descriptions for your video analysis."
        )
        
    video_tag_instructions = ""
    if target_tags:
        target_tags_str = ", ".join(f"'{t}'" for t in target_tags)
        video_tag_instructions += (
            f"Target tags criteria to evaluate against: [{target_tags_str}]. "
            "If the video matches any of these target tags/categories, assign them with appropriate confidence scores (0.0 - 1.0). "
        )
    if tag_format == "flat":
        video_tag_instructions += "\nTag format: output concise keywords in 'tag' field, category 'general'."
    elif tag_format == "prefixed":
        video_tag_instructions += "\nTag format: populate tag using 'category:tag' naming convention."
    else:
        video_tag_instructions += (
            "\nTag format: populate 'tag', explicit 'category' "
            "(e.g. content_type, event, objects, scene, people, mood), and 'confidence' (0.0 to 1.0). "
            "Also classify high-level 'content_type' (documents, social, nature, animals, screenshots, family, other)."
        )

    people_str = format_people_prompt_section(faces_data, is_video=True)
    prompt = build_vision_prompt(
        media_type="video",
        context_str=transcription_prompt,
        people_str=people_str,
        tag_instructions=video_tag_instructions,
        custom_template=custom_prompt
    )
    
    user_content = [{"type": "text", "text": prompt}]
    
    for idx in sampled_indices:
        frame_path = frames[idx]
        second = idx
        mins = second // 60
        secs = second % 60
        timestamp_str = f"{mins:02d}:{secs:02d}"
        
        base64_img = encode_image_to_base64(frame_path)
        
        user_content.append({
            "type": "text",
            "text": f"Frame at {timestamp_str}:"
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })
        
    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    res = call_local_model_api(messages, VideoAnalysis)
    res.transcription = transcription or ""
    res.transcription_ru = transcription_ru or ""
    return res

def analyze_duplicates_local(image_paths: List[Path]) -> Any:
    """Analyze burst photo group for defects and select the best photo via LM Studio."""
    from src.utils.gemini import GroupDuplicateAnalysis
    
    user_content = []
    
    prompt = (
        "Before you is a series of similar photos taken sequentially (burst shooting). "
        "Compare them against each other and determine the degree/weight of changes (changes_weight). "
        "Perform a technical defect audit of each frame for: "
        "- motion blur (has_motion_blur)\n"
        "- closed eyes (has_closed_eyes)\n"
        "- defocus (has_defocus)\n"
        "- bad exposure (has_bad_exposure)\n"
        "Determine the filename of the best frame (best_image_name) and provide detailed reasoning for the selection. "
        "Fill all main text fields in English, and provide full Russian translations in the corresponding *_ru fields "
        "so that the output JSON supports searching in both English and Russian."
    )
    user_content.append({"type": "text", "text": prompt})
    
    for path in image_paths:
        base64_img = encode_image_to_base64(path)
        user_content.append({"type": "text", "text": f"Filename: {path.name}"})
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })
        
    messages = [
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    return call_local_model_api(messages, GroupDuplicateAnalysis)
