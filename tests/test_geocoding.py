import pytest

from core.utils.geocoding import Coordinates, geocode_address


class DummyResponse:
    """Simple response object for monkeypatched requests."""

    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):  # pragma: no cover - nothing to do
        pass

    def json(self):
        return self._json


def test_geocode_yverdon(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=10):
        assert params["q"] == "Yverdon-les-Bains, Switzerland"
        return DummyResponse([{"lat": "46.78", "lon": "6.64"}])

    monkeypatch.setattr("core.utils.geocoding.requests.get", fake_get)
    coords = geocode_address("Yverdon-les-Bains, Switzerland")
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 46 <= lat <= 47
    assert 6 <= lon <= 7


def test_geocode_invalid_address(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=10):
        return DummyResponse([])

    monkeypatch.setattr("core.utils.geocoding.requests.get", fake_get)
    with pytest.raises(ValueError):
        geocode_address("Invalid Address, Nowhere")


def test_geocode_empty_address(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=10):
        return DummyResponse([])

    monkeypatch.setattr("core.utils.geocoding.requests.get", fake_get)
    with pytest.raises(ValueError):
        geocode_address("")


def test_geocode_special_characters(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=10):
        assert params["q"] == "Café de Flore, Paris, France"
        return DummyResponse([{"lat": "48.85", "lon": "2.33"}])

    monkeypatch.setattr("core.utils.geocoding.requests.get", fake_get)
    coords = geocode_address("Café de Flore, Paris, France")
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 48 <= lat <= 49
    assert 2 <= lon <= 3


def test_geocode_long_address(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=10):
        assert (
            params["q"]
            == "1600 Amphitheatre Parkway, Mountain View, CA 94043, United States"
        )
        return DummyResponse([{"lat": "37.42", "lon": "-122.08"}])

    monkeypatch.setattr("core.utils.geocoding.requests.get", fake_get)
    coords = geocode_address(
        "1600 Amphitheatre Parkway, Mountain View, CA 94043, United States"
    )
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 37 <= lat <= 38
    assert -123 <= lon <= -121


def test_geocode_address_with_zip(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=10):
        assert params["q"] == "10001, New York, NY, USA"
        return DummyResponse([{"lat": "40.75", "lon": "-73.99"}])

    monkeypatch.setattr("core.utils.geocoding.requests.get", fake_get)
    coords = geocode_address("10001, New York, NY, USA")
    assert isinstance(coords, Coordinates)
    lat, lon = coords.lat, coords.lon
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert 40 <= lat <= 41
    assert -74 <= lon <= -73
