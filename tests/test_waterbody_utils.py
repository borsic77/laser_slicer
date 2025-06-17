import os

import pytest
from shapely.geometry import Polygon

from core.utils.waterbody import (
    _is_relation_bbox_too_large,
    _element_to_polygon,
    fetch_waterbody_polygon_osmnx,
)


class FakeResponse:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_is_relation_bbox_too_large_small(monkeypatch):
    def fake_post(url, data=None, timeout=30):
        json_data = {
            "elements": [
                {"bounds": {"minlat": 0.0, "minlon": 0.0, "maxlat": 1.0, "maxlon": 1.0}}
            ]
        }
        return FakeResponse(json_data)

    monkeypatch.setattr("core.utils.waterbody.requests.post", fake_post)
    assert _is_relation_bbox_too_large(1, max_deg=2.0) is False


def test_is_relation_bbox_too_large_big(monkeypatch):
    def fake_post(url, data=None, timeout=30):
        json_data = {
            "elements": [
                {"bounds": {"minlat": 0.0, "minlon": 0.0, "maxlat": 3.0, "maxlon": 3.0}}
            ]
        }
        return FakeResponse(json_data)

    monkeypatch.setattr("core.utils.waterbody.requests.post", fake_post)
    assert _is_relation_bbox_too_large(1, max_deg=2.0) is True


def test_element_to_polygon_way():
    element = {
        "type": "way",
        "geometry": [
            {"lon": 0, "lat": 0},
            {"lon": 1, "lat": 0},
            {"lon": 1, "lat": 1},
            {"lon": 0, "lat": 1},
            {"lon": 0, "lat": 0},
        ],
    }
    poly = _element_to_polygon(element)
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(1.0)


def test_element_to_polygon_relation():
    element = {
        "type": "relation",
        "members": [
            {
                "role": "outer",
                "geometry": [
                    {"lon": 0, "lat": 0},
                    {"lon": 2, "lat": 0},
                    {"lon": 2, "lat": 2},
                    {"lon": 0, "lat": 2},
                    {"lon": 0, "lat": 0},
                ],
            },
            {
                "role": "outer",
                "geometry": [
                    {"lon": 3, "lat": 3},
                    {"lon": 4, "lat": 3},
                    {"lon": 4, "lat": 4},
                    {"lon": 3, "lat": 4},
                    {"lon": 3, "lat": 3},
                ],
            },
        ],
    }
    poly = _element_to_polygon(element)
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(4.0)


def test_fetch_waterbody_polygon_osmnx(monkeypatch, tmp_path):
    xml = """<?xml version='1.0' encoding='UTF-8'?>
    <osm version='0.6' generator='Overpass API'>
      <node id='1' lat='0' lon='0'/>
      <node id='2' lat='0' lon='1'/>
      <node id='3' lat='1' lon='1'/>
      <node id='4' lat='1' lon='0'/>
      <way id='10'>
        <nd ref='1'/>
        <nd ref='2'/>
        <nd ref='3'/>
        <nd ref='4'/>
        <nd ref='1'/>
      </way>
      <relation id='100'>
        <member type='way' ref='10' role='outer'/>
        <tag k='type' v='multipolygon'/>
      </relation>
    </osm>"""

    def fake_get(url):
        return FakeResponse(content=xml.encode("utf-8"))

    monkeypatch.setattr("core.utils.waterbody.requests.get", fake_get)

    poly = fetch_waterbody_polygon_osmnx(100)
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(1.0)

