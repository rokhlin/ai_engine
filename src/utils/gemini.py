import io
import json
import time
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types
from src import config

# --- Pydantic Data Schemas for Structured Output ---

class TagItem(BaseModel):
    tag: str = Field(description="Tag name or label")
    category: Optional[str] = Field(default="general", description="Category of the tag (e.g. content_type, event, objects, scene, people, mood)")
    confidence: Optional[float] = Field(default=1.0, description="Confidence score between 0.0 and 1.0")

    @classmethod
    def from_any(cls, val: Any) -> "TagItem":
        if isinstance(val, TagItem):
            return val
        if isinstance(val, str):
            val_clean = val.strip()
            if ":" in val_clean:
                cat, t = val_clean.split(":", 1)
                return cls(tag=t.strip(), category=cat.strip().lower(), confidence=1.0)
            return cls(tag=val_clean, category="general", confidence=1.0)
        if isinstance(val, dict):
            tag_name = str(val.get("tag") or val.get("name") or "").strip()
            cat = str(val.get("category", "general")).strip().lower()
            try:
                conf = float(val.get("confidence", 1.0))
            except (ValueError, TypeError):
                conf = 1.0
            return cls(tag=tag_name, category=cat, confidence=conf)
        return cls(tag=str(val).strip(), category="general", confidence=1.0)


class PhotoAnalysis(BaseModel):
    summary: str = Field(description="Short summary of the frame content in English")
    summary_ru: str = Field(description="Short summary of the frame content translated into Russian")
    description: str = Field(description="Detailed semantic description of the scene in English")
    description_ru: str = Field(description="Detailed semantic description of the scene translated into Russian")
    environment: Literal["indoor", "outdoor", "unknown"] = Field(description="Type of environment")
    lighting: str = Field(description="Lighting characteristics (natural, studio, dim, etc.) in English")
    lighting_ru: str = Field(description="Lighting characteristics in Russian")
    weather: Optional[str] = Field(default=None, description="Weather if photo was taken outdoors in English")
    weather_ru: Optional[str] = Field(default=None, description="Weather if photo was taken outdoors in Russian")
    time_of_day: str = Field(description="Time of day (morning, day, evening, night) in English")
    time_of_day_ru: str = Field(description="Time of day in Russian")
    content_type: Optional[str] = Field(default="other", description="High-level content classification: documents, social, nature, animals, screenshots, family, other")
    tags: List[TagItem] = Field(default_factory=list, description="Categorized semantic tags matching requested format")
    ocr_text: Optional[str] = Field(default=None, description="Recognized text in the image (OCR) if present")
    exif_analysis: Optional[str] = Field(default=None, description="Expert analysis based on EXIF metadata in English")
    exif_analysis_ru: Optional[str] = Field(default=None, description="Expert analysis based on EXIF metadata in Russian")

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            items = [t.strip() for t in v.split(",") if t.strip()]
            return [TagItem.from_any(t) for t in items if t]
        if isinstance(v, list):
            res = []
            for item in v:
                t = TagItem.from_any(item)
                if t.tag:
                    res.append(t)
            return res
        return []


class TimelineEvent(BaseModel):
    timestamp_start: str = Field(description="Start of event in MM:SS format")
    timestamp_end: str = Field(description="End of event in MM:SS format")
    activity: str = Field(description="Description of activity/event in English")
    activity_ru: str = Field(description="Description of activity/event translated into Russian")


class VideoAnalysis(BaseModel):
    summary: str = Field(description="General description of video plot in English")
    summary_ru: str = Field(description="General description of video plot translated into Russian")
    transcription: str = Field(description="Transcription of speech and key background sounds in English")
    transcription_ru: str = Field(description="Transcription of speech and key background sounds translated into Russian")
    timeline_events: List[TimelineEvent] = Field(description="Segmented timeline of events with activity labels")
    content_type: Optional[str] = Field(default="other", description="High-level content classification: documents, social, nature, animals, screenshots, family, other")
    tags: List[TagItem] = Field(default_factory=list, description="Categorized semantic tags matching requested format")

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v):
        if not v:
            return []
        if isinstance(v, str):
            items = [t.strip() for t in v.split(",") if t.strip()]
            return [TagItem.from_any(t) for t in items if t]
        if isinstance(v, list):
            res = []
            for item in v:
                t = TagItem.from_any(item)
                if t.tag:
                    res.append(t)
            return res
        return []


class DefectAnalysis(BaseModel):
    filename: str = Field(description="Image file name")
    has_motion_blur: bool = Field(description="Presence of motion blur")
    has_closed_eyes: bool = Field(description="Presence of closed eyes")
    has_defocus: bool = Field(description="Presence of defocus/blur")
    has_bad_exposure: bool = Field(description="Presence of bad exposure (over/underexposed)")
    details: str = Field(description="Details on defects or overall image quality in English")
    details_ru: str = Field(description="Details on defects or overall image quality translated into Russian")


class GroupDuplicateAnalysis(BaseModel):
    changes_weight: Literal["negligible", "low", "medium", "high"] = Field(
        description="Degree of changes between frames in burst (negligible, low, medium, high)"
    )
    defects: List[DefectAnalysis] = Field(description="Defect analysis for each frame in the burst")
    best_image_name: str = Field(description="Filename of the best frame in the burst")
    reasoning: str = Field(description="Reasoning for selecting the best frame in English")
    reasoning_ru: str = Field(description="Reasoning for selecting the best frame translated into Russian")


import threading
from collections import deque

# --- Thread-Safe Rate Limiter for Concurrent API Calls ---

class GeminiRateLimiter:
    """Thread-safe sliding-window rate limiter for Gemini API requests."""
    def __init__(self, rpm_limit: int = 15):
        self.rpm_limit = max(1, rpm_limit)
        self.lock = threading.Lock()
        self.timestamps = deque()

    def acquire(self):
        """Block until a request slot is available under RPM limits."""
        while True:
            with self.lock:
                now = time.time()
                # Purge timestamps older than 60s
                while self.timestamps and now - self.timestamps[0] > 60.0:
                    self.timestamps.popleft()
                
                if len(self.timestamps) < self.rpm_limit:
                    self.timestamps.append(now)
                    return
                
                # Calculate required sleep until the oldest request slot frees up
                oldest = self.timestamps[0]
                sleep_needed = max(0.1, 60.0 - (now - oldest) + 0.05)
            time.sleep(sleep_needed)

_rate_limiter = GeminiRateLimiter(rpm_limit=config.RPM_LIMIT)
_client = None
_client_lock = threading.Lock()

def get_gemini_client() -> genai.Client:
    """Initialize singleton Google GenAI client in a thread-safe manner."""
    global _client
    with _client_lock:
        if _client is None:
            if not config.GEMINI_API_KEY:
                raise ValueError("Critical error: GEMINI_API_KEY environment variable is not set.")
            _client = genai.Client(api_key=config.GEMINI_API_KEY)
        return _client

def prepare_image_bytes(image_path: Path) -> bytes:
    """Scale image and convert to JPEG bytes."""
    try:
        with Image.open(image_path) as img:
            # Convert modes (e.g. RGBA, CMYK, Palette) to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Scale to IMAGE_MAX_SIZE along the larger dimension
            w, h = img.size
            max_sz = config.IMAGE_MAX_SIZE
            if w > max_sz or h > max_sz:
                if w > h:
                    new_w = max_sz
                    new_h = int(h * (max_sz / w))
                else:
                    new_h = max_sz
                    new_w = int(w * (max_sz / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            # Save to memory buffer
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:
        raise RuntimeError(f"Failed to prepare image {image_path.name}: {e}") from e

# --- Gemini API Call Methods ---

def format_people_prompt_section(
    faces_data: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]], 
    is_video: bool = False
) -> str:
    """Format prompt instructions for recognized/named people without technical coordinates in output."""
    if not faces_data:
        return ""
    
    if not is_video:
        # Photo faces list: [{"name": ..., "bbox": [...], "face_id": ...}, ...]
        if not isinstance(faces_data, list):
            return ""
            
        named_faces = []
        for f in faces_data:
            if not isinstance(f, dict):
                continue
            name = (f.get("name") or "").strip()
            face_id = (f.get("face_id") or "").strip()
            if name and name.lower() not in ("unknown", "unnamed") and not name.lower().startswith(("face_", "face-", "person_")):
                bbox = f.get("bbox")
                named_faces.append({"name": name, "bbox": bbox})
            elif name and name != face_id and not face_id.startswith(("manual_", "face_manual_")):
                bbox = f.get("bbox")
                named_faces.append({"name": name, "bbox": bbox})
                
        if not named_faces:
            return ""
            
        people_lines = []
        for idx, item in enumerate(named_faces, 1):
            bbox_desc = f" (bounding box: {item['bbox']})" if item.get("bbox") else ""
            people_lines.append(f'- Person {idx}: "{item["name"]}"{bbox_desc}')
            
        people_text = "\n".join(people_lines)
        return (
            f"Recognized individuals present in this photo:\n{people_text}\n\n"
            "MANDATORY INSTRUCTIONS FOR RECOGNIZED PEOPLE:\n"
            "1. You MUST explicitly mention each identified person by name in your scene summary ('summary', 'summary_ru') "
            "and detailed description ('description', 'description_ru').\n"
            "2. Use the bounding boxes provided above only to visually match which individual in the photo is which named person. "
            "CRITICAL: Do NOT mention raw coordinates, numbers, or bounding boxes anywhere in the summary, description, or tags.\n"
            "3. Describe people naturally by their visual appearance, clothing, posture, and activities (e.g. 'Alice is smiling in a blue dress while Bob stands beside her...').\n"
            "4. If distinguishing who is who is ambiguous or uncertain, describe all identified individuals naturally together close to the original scene context (e.g. 'The photo captures Alice and Bob attending an outdoor event...').\n"
            "5. Include each recognized person as a tag with category 'people' and confidence 1.0 (e.g. tag: 'Alice', category: 'people')."
        )
    else:
        # Video faces: dict {face_id: {"name": ..., "intervals": [...]}} or list
        named_intervals = []
        if isinstance(faces_data, dict):
            for fid, info in faces_data.items():
                if not isinstance(info, dict):
                    continue
                name = (info.get("name") or "").strip()
                if name and name.lower() not in ("unknown", "unnamed") and not name.lower().startswith(("face_", "face-", "person_")):
                    intervals = info.get("intervals") or []
                    named_intervals.append({"name": name, "intervals": intervals})
                elif name and name != fid:
                    intervals = info.get("intervals") or []
                    named_intervals.append({"name": name, "intervals": intervals})
        elif isinstance(faces_data, list):
            for item in faces_data:
                if isinstance(item, dict):
                    name = (item.get("name") or "").strip()
                    fid = (item.get("face_id") or "").strip()
                    if name and name.lower() not in ("unknown", "unnamed") and not name.lower().startswith(("face_", "face-", "person_")):
                        named_intervals.append({"name": name, "intervals": item.get("intervals") or item.get("time_intervals") or []})
                    elif name and name != fid:
                        named_intervals.append({"name": name, "intervals": item.get("intervals") or item.get("time_intervals") or []})

        if not named_intervals:
            return ""
            
        people_lines = []
        for idx, item in enumerate(named_intervals, 1):
            int_desc = f" (visible during timecodes: {', '.join(str(i) for i in item['intervals'])})" if item.get("intervals") else ""
            people_lines.append(f'- Person {idx}: "{item["name"]}"{int_desc}')
            
        people_text = "\n".join(people_lines)
        return (
            f"Recognized individuals appearing in this video:\n{people_text}\n\n"
            "MANDATORY INSTRUCTIONS FOR RECOGNIZED PEOPLE:\n"
            "1. You MUST explicitly mention each identified person by name in your summary ('summary', 'summary_ru'), "
            "description ('description', 'description_ru'), and in timeline_events activity descriptions corresponding to their appearance timecodes.\n"
            "2. Do NOT output raw tracking IDs or numbers. Describe what each individual is doing naturally.\n"
            "3. If distinguishing which person is performing a specific action is ambiguous or uncertain, describe all identified individuals naturally together close to the original scene context.\n"
            "4. Include each recognized person as a tag with category 'people' and confidence 1.0 (e.g. tag: 'Alice', category: 'people')."
        )


def build_vision_prompt(
    media_type: str,
    context_str: str = "",
    people_str: str = "",
    tag_instructions: str = "",
    custom_template: Optional[str] = None
) -> str:
    """Compile final prompt using template placeholders or graceful appending."""
    import re
    template = custom_template or getattr(config, "VISION_PROMPT_TEMPLATE", None) or config.DEFAULT_VISION_PROMPT_TEMPLATE
    
    has_media_type = "{media_type}" in template
    has_context = "{context}" in template
    has_people = "{people}" in template
    has_tags = "{tag_instructions}" in template
    
    media_type_desc = "photo" if media_type == "photo" else "video file plot"
    
    prompt = template
    if has_media_type:
        prompt = prompt.replace("{media_type}", media_type_desc)
    if has_context:
        prompt = prompt.replace("{context}", context_str.strip())
    if has_people:
        prompt = prompt.replace("{people}", people_str.strip())
    if has_tags:
        prompt = prompt.replace("{tag_instructions}", tag_instructions.strip())
        
    extra_parts = []
    if not has_context and context_str.strip():
        extra_parts.append(context_str.strip())
    if not has_people and people_str.strip():
        extra_parts.append(people_str.strip())
    if not has_tags and tag_instructions.strip():
        extra_parts.append(tag_instructions.strip())
        
    if extra_parts:
        prompt = prompt.strip() + "\n\n" + "\n\n".join(extra_parts)
        
    prompt = re.sub(r'\n{3,}', '\n\n', prompt).strip()
    return prompt


def analyze_photo(
    image_path: Path, 
    exif_data: Optional[Dict[str, Any]] = None,
    target_tags: Optional[List[str]] = None,
    tag_format: Optional[str] = "categorized",
    faces_data: Optional[List[Dict[str, Any]]] = None,
    custom_prompt: Optional[str] = None
) -> PhotoAnalysis:
    """Send photo for analysis to Gemini API or Local model, incorporating EXIF metadata, recognized faces, and optional target tags/formatting."""
    if config.MODEL_PROVIDER == "local":
        from src.utils.local_model import analyze_photo_local
        return analyze_photo_local(
            image_path, 
            exif_data=exif_data, 
            target_tags=target_tags, 
            tag_format=tag_format,
            faces_data=faces_data,
            custom_prompt=custom_prompt
        )
        
    client = get_gemini_client()
    image_bytes = prepare_image_bytes(image_path)
    
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
    
    _rate_limiter.acquire()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PhotoAnalysis,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
    )
    
    return PhotoAnalysis.model_validate_json(response.text)

def analyze_video(
    video_path: Path, 
    frames: Optional[List[Path]] = None,
    transcription: Optional[str] = None,
    transcription_ru: Optional[str] = None,
    target_tags: Optional[List[str]] = None,
    tag_format: Optional[str] = "categorized",
    faces_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    custom_prompt: Optional[str] = None
) -> VideoAnalysis:
    """Perform semantic analysis on video file via Gemini File API or Local model."""
    if config.MODEL_PROVIDER == "local":
        from src.utils.local_model import analyze_video_local
        return analyze_video_local(
            video_path, 
            frames=frames, 
            transcription=transcription, 
            transcription_ru=transcription_ru,
            target_tags=target_tags,
            tag_format=tag_format,
            faces_data=faces_data,
            custom_prompt=custom_prompt
        )
        
    client = get_gemini_client()
    
    print(f"Uploading video {video_path.name} to Google GenAI File API...")
    _rate_limiter.acquire()
    uploaded_file = client.files.upload(file=str(video_path))
    
    try:
        # Await video processing on Google API side
        print("Waiting for video processing on Google API side...")
        while True:
            file_info = client.files.get(name=uploaded_file.name)
            state = getattr(file_info, "state", "PROCESSING")
            if state == "ACTIVE":
                print("Video is ready for analysis.")
                break
            elif state in ("FAILED", "ERROR"):
                error_msg = getattr(file_info.error, "message", "Unknown error")
                raise RuntimeError(f"Video processing on API side failed: {error_msg}")
            
            time.sleep(10)
            
        transcription_prompt = ""
        if transcription:
            transcription_prompt = (
                "We have already transcribed the speech/audio of this video for you:\n"
                f"Transcription (English): {transcription}\n"
            )
            if transcription_ru:
                transcription_prompt += f"Transcription (Russian): {transcription_ru}\n"
            transcription_prompt += (
                "Please use this transcription to help understand the context, dialogue, and plot descriptions. "
                "You should output this transcription or a refined version of it in the transcription/transcription_ru fields."
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
        
        _rate_limiter.acquire()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VideoAnalysis,
                temperature=0.1,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            )
        )
        
        return VideoAnalysis.model_validate_json(response.text)
        
    finally:
        # Always delete the file from File API after processing
        print(f"Deleting temporary video file {uploaded_file.name} from cloud...")
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception as e:
            print(f"Failed to delete file {uploaded_file.name} from File API: {e}")

def analyze_duplicates(image_paths: List[Path]) -> GroupDuplicateAnalysis:
    """Batch analysis of burst shot duplicates via Gemini API or Local model."""
    if config.MODEL_PROVIDER == "local":
        from src.utils.local_model import analyze_duplicates_local
        return analyze_duplicates_local(image_paths)
        
    client = get_gemini_client()
    
    contents = []
    # Add all images to a single request
    for path in image_paths:
        img_bytes = prepare_image_bytes(path)
        # Associate metadata filename with file
        contents.append(f"Filename: {path.name}")
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
        
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
    contents.append(prompt)
    
    _rate_limiter.acquire()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GroupDuplicateAnalysis,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
    )
    
    return GroupDuplicateAnalysis.model_validate_json(response.text)

