import pytest
from src.utils import video
from src import workers

def test_group_seconds_into_intervals():
    # Empty list
    assert video.group_seconds_into_intervals([]) == []
    assert workers.group_seconds_into_intervals([]) == []

    # Single point
    assert video.group_seconds_into_intervals([5]) == [("00:05", "00:05")]

    # Continuous range
    assert video.group_seconds_into_intervals([1, 2, 3]) == [("00:01", "00:03")]

    # Small gap (<=2 seconds) groups together
    assert video.group_seconds_into_intervals([1, 3]) == [("00:01", "00:03")]

    # Large gap splits
    intervals = video.group_seconds_into_intervals([1, 2, 10, 11, 75])
    assert intervals == [("00:01", "00:02"), ("00:10", "00:11"), ("01:15", "01:15")]

def test_format_timestamp():
    assert video.format_timestamp(0) == "00:00"
    assert video.format_timestamp(65) == "01:05"
    assert video.format_timestamp(3600) == "60:00"
