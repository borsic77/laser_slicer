import json
import requests

import pytest
from shapely.geometry import Point, Polygon

from core.services.osm_water_service import OSMWaterService


def test_fetch_water_polygon_large_lake(monkeypatch):
    # Fake Overpass response for a square lake around (8,45)
    fake_response = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "geometry": [
                    {"lat": 45.0, "lon": 8.0},
                    {"lat": 45.0, "lon": 8.1},
                    {"lat": 45.1, "lon": 8.1},
                    {"lat": 45.1, "lon": 8.0},
                    {"lat": 45.0, "lon": 8.0},
                ],
                "tags": {"natural": "water"},
            }
        ]
    }

    class DummyResp:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_post(self, url, data=None, timeout=30):
        return DummyResp(fake_response)

    monkeypatch.setattr(requests.Session, "post", fake_post)

    service = OSMWaterService(radii=[100, 200])
    poly = service.fetch_water_polygon(45.05, 8.05)
    assert isinstance(poly, Polygon)
    assert poly.contains(Point(8.05, 45.05))
