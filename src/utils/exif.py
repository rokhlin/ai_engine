import reverse_geocoder as rg
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

def _dms_to_decimal(dms_tuple, ref_dir) -> Optional[float]:
    """Convert coordinates from Degrees/Minutes/Seconds (DMS) to decimal format."""
    try:
        # DMS in Pillow can be stored as a 3-element tuple (Rational or float/int)
        degrees = float(dms_tuple[0])
        minutes = float(dms_tuple[1])
        seconds = float(dms_tuple[2])
        
        val = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref_dir in ('S', 'W'):
            val = -val
        return val
    except Exception:
        return None

def extract_gps_coords(exif_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Extract latitude and longitude from EXIF data."""
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None, None
    
    # Decode GPSInfo tags
    gps_decoded = {}
    for tag, val in gps_info.items():
        tag_name = GPSTAGS.get(tag, tag)
        gps_decoded[tag_name] = val
        
    lat_val = gps_decoded.get("GPSLatitude")
    lat_ref = gps_decoded.get("GPSLatitudeRef")
    lon_val = gps_decoded.get("GPSLongitude")
    lon_ref = gps_decoded.get("GPSLongitudeRef")
    
    if not (lat_val and lat_ref and lon_val and lon_ref):
        return None, None
        
    lat = _dms_to_decimal(lat_val, lat_ref)
    lon = _dms_to_decimal(lon_val, lon_ref)
    return lat, lon

def get_offline_address(lat: float, lon: float) -> Optional[Dict[str, str]]:
    """Local (offline) reverse geocoding by coordinates."""
    try:
        # rg.search accepts a list of coordinate tuples
        res = rg.search((lat, lon))
        if res:
            info = res[0]
            return {
                "city": info.get("name", ""),
                "region": info.get("admin1", ""),
                "country": info.get("cc", "")
            }
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None

def read_photo_metadata(file_path: Path) -> Dict[str, Any]:
    """Read photo EXIF metadata and enrich with reverse geocoding."""
    meta = {
        "datetime": None,
        "camera_make": None,
        "camera_model": None,
        "iso": None,
        "exposure_time": None,
        "f_number": None,
        "gps": None,
        "location": None
    }
    
    try:
        with Image.open(file_path) as img:
            exif = img._getexif()
            if not exif:
                return meta
                
            decoded = {}
            for tag, val in exif.items():
                tag_name = TAGS.get(tag, tag)
                decoded[tag_name] = val
                
            # Extract date/time
            # Priority: DateTimeOriginal -> DateTimeDigitized -> DateTime
            dt_str = decoded.get("DateTimeOriginal") or decoded.get("DateTimeDigitized") or decoded.get("DateTime")
            if dt_str:
                meta["datetime"] = str(dt_str)
                
            meta["camera_make"] = str(decoded.get("Make")) if decoded.get("Make") else None
            meta["camera_model"] = str(decoded.get("Model")) if decoded.get("Model") else None
            meta["iso"] = int(decoded.get("ISOSpeedRatings")) if decoded.get("ISOSpeedRatings") else None
            
            # Exposure time in seconds (can be a Fraction)
            exp = decoded.get("ExposureTime")
            if exp:
                meta["exposure_time"] = f"{float(exp):.5f}" if float(exp) < 1.0 else str(exp)
                
            f_num = decoded.get("FNumber")
            if f_num:
                meta["f_number"] = f"f/{float(f_num):.1f}"
                
            # GPS
            lat, lon = extract_gps_coords(decoded)
            if lat is not None and lon is not None:
                meta["gps"] = {"latitude": lat, "longitude": lon}
                # Local geocoding
                loc = get_offline_address(lat, lon)
                if loc:
                    meta["location"] = loc
                    
    except Exception as e:
        print(f"Error reading EXIF from {file_path.name}: {e}")
        
    return meta

