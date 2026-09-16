import pytest
from src.utils import exif

def test_dms_to_decimal():
    # 30 deg, 15 min, 50 sec North
    dms = (30, 15, 50)
    decimal = exif._dms_to_decimal(dms, "N")
    assert decimal is not None
    assert pytest.approx(decimal, 0.0001) == 30.263888
    
    # 30 deg, 15 min, 50 sec South (should be negative)
    decimal_s = exif._dms_to_decimal(dms, "S")
    assert decimal_s is not None
    assert pytest.approx(decimal_s, 0.0001) == -30.263888
    
    # Invalid values
    assert exif._dms_to_decimal(None, "N") is None
    assert exif._dms_to_decimal((30, 15), "N") is None

def test_extract_gps_coords():
    # Mock Pillow EXIF structure for GPSInfo
    exif_data = {
        "GPSInfo": {
            0: b"\x02\x02\x00\x00", # GPSVersionID
            1: "N",                 # GPSLatitudeRef
            2: (55.0, 45.0, 30.0),  # GPSLatitude
            3: "E",                 # GPSLongitudeRef
            4: (37.0, 35.0, 15.0)   # GPSLongitude
        }
    }
    
    lat, lon = exif.extract_gps_coords(exif_data)
    assert lat is not None
    assert lon is not None
    assert pytest.approx(lat, 0.0001) == 55.75833
    assert pytest.approx(lon, 0.0001) == 37.5875
    
    # Test without GPS
    lat2, lon2 = exif.extract_gps_coords({})
    assert lat2 is None
    assert lon2 is None
