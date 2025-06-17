import os
import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from core.utils.geocoding import (
    _parse_bounds,
    get_transformer_for_coords,
    compute_utm_bounds_from_wgs84,
)


def test_parse_bounds():
    data = {
        "lon_min": "6.5",
        "lat_min": "46.7",
        "lon_max": "6.7",
        "lat_max": "46.9",
    }
    result = _parse_bounds(data)
    assert result == (6.5, 46.7, 6.7, 46.9)
    assert all(isinstance(v, float) for v in result)


def test_get_transformer_for_coords():
    transformer = get_transformer_for_coords(12.0, 55.0)
    assert transformer.target_crs.to_epsg() == 32633
    south = get_transformer_for_coords(-72.0, -15.0)
    assert south.target_crs.to_epsg() == 32719


def test_compute_utm_bounds_from_wgs84():
    lon_min, lat_min, lon_max, lat_max = 6.5, 46.7, 6.7, 46.9
    bounds = compute_utm_bounds_from_wgs84(
        lon_min,
        lat_min,
        lon_max,
        lat_max,
        6.6,
        46.8,
    )
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]
    assert bounds[0] == pytest.approx(308878, rel=0.01)
    assert bounds[1] == pytest.approx(5174862, rel=0.01)
    assert bounds[2] == pytest.approx(324818, rel=0.01)
    assert bounds[3] == pytest.approx(5196619, rel=0.01)
