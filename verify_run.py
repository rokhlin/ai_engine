import os
import sys
import shutil
import time
import json
from pathlib import Path
from PIL import Image
from unittest.mock import patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Set test environment variables BEFORE importing src.config
os.environ["GEMINI_API_KEY"] = "MOCK_API_KEY"
os.environ["INPUT_FOLDERS"] = "media_input"
os.environ["OUTPUT_FOLDER"] = "media_output"
os.environ["DB_PATH"] = "media_output/catalog_history.db"

from src import config
config.INPUT_FOLDERS = [Path("media_input")]
config.OUTPUT_FOLDER = Path("media_output")
config.DB_PATH = Path("media_output/catalog_history.db")

from src import database
from src import workers
from src.utils import gemini

# --- Mocks for Gemini API ---

class MockPhotoAnalysis:
    def __init__(self):
        self.summary = "A nice red frame"
        self.summary_ru = "Красный кадр"
        self.description = "Detailed description of the test image"
        self.description_ru = "Подробное описание тестового изображения"
        self.environment = "indoor"
        self.lighting = "natural"
        self.lighting_ru = "естественное"
        self.weather = None
        self.weather_ru = None
        self.time_of_day = "day"
        self.time_of_day_ru = "день"
        self.ocr_text = "Test Text"
        self.exif_analysis = "Test EXIF analysis"
        self.exif_analysis_ru = "Тестовый EXIF анализ"
        
    def model_dump(self):
        return {
            "summary": self.summary,
            "summary_ru": self.summary_ru,
            "description": self.description,
            "description_ru": self.description_ru,
            "environment": self.environment,
            "lighting": self.lighting,
            "lighting_ru": self.lighting_ru,
            "weather": self.weather,
            "weather_ru": self.weather_ru,
            "time_of_day": self.time_of_day,
            "time_of_day_ru": self.time_of_day_ru,
            "ocr_text": self.ocr_text,
            "exif_analysis": self.exif_analysis,
            "exif_analysis_ru": self.exif_analysis_ru
        }

class MockDefectAnalysis:
    def __init__(self, filename):
        self.filename = filename
        self.has_motion_blur = False
        self.has_closed_eyes = False
        self.has_defocus = False
        self.has_bad_exposure = False
        self.details = "Excellent quality"
        self.details_ru = "Отличное качество"
        
    def model_dump(self):
        return {
            "filename": self.filename,
            "has_motion_blur": self.has_motion_blur,
            "has_closed_eyes": self.has_closed_eyes,
            "has_defocus": self.has_defocus,
            "has_bad_exposure": self.has_bad_exposure,
            "details": self.details,
            "details_ru": self.details_ru
        }

class MockGroupDuplicateAnalysis:
    def __init__(self, best_image_name):
        self.changes_weight = "negligible"
        self.best_image_name = best_image_name
        self.reasoning = "Frame is sharp, no closed eyes"
        self.reasoning_ru = "Кадр чёткий, закрытых глаз нет"
        self.defects = [MockDefectAnalysis("photo1.jpg"), MockDefectAnalysis("photo2.jpg")]
        
    def model_dump(self):
        return {
            "changes_weight": self.changes_weight,
            "defects": [d.model_dump() for d in self.defects],
            "best_image_name": self.best_image_name,
            "reasoning": self.reasoning,
            "reasoning_ru": self.reasoning_ru
        }

def mock_analyze_photo(image_path, exif_data=None):
    return MockPhotoAnalysis()

def mock_analyze_duplicates(image_paths):
    # Make the first one in the list the best one
    best_name = image_paths[0].name
    return MockGroupDuplicateAnalysis(best_name)

# --- Prepare test files ---

def create_test_media():
    input_dir = Path("media_input")
    input_dir.mkdir(exist_ok=True)
    
    # 1. photo1.jpg (red, time t)
    img1 = Image.new("RGB", (100, 100), color="red")
    exif1 = img1.getexif()
    exif1[36867] = "2023:08:20 14:00:00" # DateTimeOriginal
    img1.save(input_dir / "photo1.jpg", exif=exif1)
    
    # 2. photo2.jpg (red, time t+2s) -> duplicate of photo1
    img2 = Image.new("RGB", (100, 100), color="red")
    exif2 = img2.getexif()
    exif2[36867] = "2023:08:20 14:00:02"
    img2.save(input_dir / "photo2.jpg", exif=exif2)
    
    # 3. photo3.jpg (blue, time t+20s) -> distinct photo
    img3 = Image.new("RGB", (100, 100), color="blue")
    exif3 = img3.getexif()
    exif3[36867] = "2023:08:20 14:00:20"
    img3.save(input_dir / "photo3.jpg", exif=exif3)

def cleanup():
    import gc
    gc.collect()
    for folder in ["media_input", "media_output"]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Cleanup warning: {e}")

def main():
    cleanup()
    create_test_media()
    
    # Patch Gemini calls
    with patch("src.utils.gemini.analyze_photo", side_effect=mock_analyze_photo), \
         patch("src.utils.gemini.analyze_duplicates", side_effect=mock_analyze_duplicates):
         
        print("=== STEP 1: First pipeline run (full sync) ===")
        workers.run_pipeline()
        
        # Verify sidecar files exist
        sidecar1 = Path("media_output/photo1.jpg.json")
        sidecar2 = Path("media_output/photo2.jpg.json")
        sidecar3 = Path("media_output/photo3.jpg.json")
        
        assert sidecar1.is_file(), "Error: Sidecar for photo1 not created"
        assert sidecar2.is_file(), "Error: Sidecar for photo2 not created"
        assert sidecar3.is_file(), "Error: Sidecar for photo3 not created"
        print("Test: All sidecar files successfully created.")
        
        # Verify duplicate_analysis block and gemini_analysis exif_analysis
        with open(sidecar1, "r", encoding="utf-8") as f:
            data1 = json.load(f)
            assert data1["duplicate_analysis"] is not None
            assert data1["duplicate_analysis"]["group_id"] == "dup_group_001"
            assert data1["gemini_analysis"].get("exif_analysis") == "Test EXIF analysis"
            print("Test: Duplicate block and EXIF analysis in photo1.jpg are correct.")
            
        with open(sidecar3, "r", encoding="utf-8") as f:
            data3 = json.load(f)
            assert data3["duplicate_analysis"] is None
            print("Test: Duplicate block in photo3.jpg is null (correct).")
            
        print("\n=== STEP 2: Second run (no changes) ===")
        # All 3 files should be skipped.
        # Patch process_photo to raise assertion error if it runs
        def error_process(*args, **kwargs):
            raise AssertionError("Files should not be re-processed!")
            
        with patch("src.workers.process_photo", side_effect=error_process):
            workers.run_pipeline()
            print("Test: Second run successfully skipped unchanged files.")
            
        print("\n=== STEP 3: Adding a new file ===")
        # Add photo4.jpg (green)
        img4 = Image.new("RGB", (100, 100), color="green")
        exif4 = img4.getexif()
        exif4[36867] = "2023:08:20 14:01:00"
        img4.save(Path("media_input/photo4.jpg"), exif=exif4)
        
        processed_files = []
        old_process_photo = workers.process_photo
        
        def spy_process_photo(photo_path, root):
            processed_files.append(photo_path.name)
            return old_process_photo(photo_path, root)
            
        with patch("src.workers.process_photo", side_effect=spy_process_photo):
            workers.run_pipeline()
            # ONLY photo4.jpg should be processed
            assert processed_files == ["photo4.jpg"], f"Error: Wrong files processed: {processed_files}"
            print("Test: Only the newly added file was processed.")
            
    print("\nVERIFICATION PASSED SUCCESSFULLY!")
    cleanup()

if __name__ == "__main__":
    main()
