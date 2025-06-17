import xml.etree.ElementTree as ET

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from core.utils.waterbody import (
    _element_to_polygon,
    _is_relation_bbox_too_large,
    fetch_waterbody_polygon_osmnx,
)


class DummyResponse:
    def __init__(self, *, json_data=None, text=""):
        self._json = json_data
        self.content = text.encode()

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_is_relation_bbox_too_large_small(monkeypatch):
    def fake_post(url, data=None, timeout=30):
        return DummyResponse(json_data={"elements": [{"bounds": {"minlat": 0, "minlon": 0, "maxlat": 0.5, "maxlon": 0.5}}]})

    monkeypatch.setattr("core.utils.waterbody.requests.post", fake_post)
    assert not _is_relation_bbox_too_large(1)


def test_is_relation_bbox_too_large_large(monkeypatch):
    def fake_post(url, data=None, timeout=30):
        return DummyResponse(json_data={"elements": [{"bounds": {"minlat": 0, "minlon": 0, "maxlat": 3.0, "maxlon": 3.0}}]})

    monkeypatch.setattr("core.utils.waterbody.requests.post", fake_post)
    assert _is_relation_bbox_too_large(1)


def test_element_to_polygon_way():
    element = {
        "type": "way",
        "geometry": [
            {"lon": 0, "lat": 0},
            {"lon": 1, "lat": 0},
            {"lon": 1, "lat": 1},
            {"lon": 0, "lat": 1},
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
                ],
            },
            {
                "role": "outer",
                "geometry": [
                    {"lon": 3, "lat": 3},
                    {"lon": 3.5, "lat": 3},
                    {"lon": 3.5, "lat": 3.5},
                    {"lon": 3, "lat": 3.5},
                ],
            },
        ],
    }
    poly = _element_to_polygon(element)
    assert isinstance(poly, Polygon)
    # Largest polygon has area 4
    assert poly.area == pytest.approx(4.0)


OSM_XML = """
<osm version='0.6'>
  <node id='1' lat='0' lon='0'/>
  <node id='2' lat='1' lon='0'/>
  <node id='3' lat='1' lon='1'/>
  <node id='4' lat='0' lon='1'/>
  <way id='10'>
    <nd ref='1'/>
    <nd ref='2'/>
    <nd ref='3'/>
    <nd ref='4'/>
    <nd ref='1'/>
    <tag k='natural' v='water'/>
  </way>
  <relation id='20'>
    <member type='way' ref='10' role='outer'/>
    <tag k='type' v='multipolygon'/>
    <tag k='natural' v='water'/>
  </relation>
</osm>
"""


def mock_features_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat"))) for n in root.iter("node")}
    coords = [nodes[nd.get("ref")] for nd in root.find("way").iter("nd")]
    poly = Polygon(coords)
    return gpd.GeoDataFrame({"geometry": [poly]})


def test_fetch_waterbody_polygon_osmnx(monkeypatch):
    def fake_get(url):
        return DummyResponse(text=OSM_XML)

    monkeypatch.setattr("core.utils.waterbody.requests.get", fake_get)
    monkeypatch.setattr("core.utils.waterbody.ox.features_from_xml", mock_features_from_xml)

    poly = fetch_waterbody_polygon_osmnx(20)
    assert isinstance(poly, Polygon)
    assert poly.area == pytest.approx(1.0)
