import pytest

from core.utils.geocoding import Coordinates, geocode_address


def test_geocode_yverdon():
    coords = geocode_address("Yverdon-les-Bains, Switzerland")
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 46 <= lat <= 47
    assert 6 <= lon <= 7


def test_geocode_invalid_address():
    with pytest.raises(ValueError):
        geocode_address("Invalid Address, Nowhere")


def test_geocode_empty_address():
    with pytest.raises(ValueError):
        geocode_address("")


def test_geocode_special_characters():
    coords = geocode_address("Café de Flore, Paris, France")
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 48 <= lat <= 49
    assert 2 <= lon <= 3


def test_geocode_long_address():
    coords = geocode_address(
        "1600 Amphitheatre Parkway, Mountain View, CA 94043, United States"
    )
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 37 <= lat <= 38
    assert -123 <= lon <= -121


def test_geocode_address_with_zip():
    coords = geocode_address("10001, New York, NY, USA")
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 40 <= lat <= 41
    assert -74 <= lon <= -73
