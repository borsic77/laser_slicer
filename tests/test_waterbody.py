import numpy as np
import pytest
from shapely.geometry import Point, Polygon

import rasterio.transform
from core.utils.waterbody import fetch_waterbody_polygon
from core.utils.slicer import generate_contours


def test_fetch_waterbody_polygon_found(monkeypatch):
    """Fetch a water polygon without real network calls."""

    # First Overpass call returns a relation id
    def fake_post(url, data=None, timeout=30):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"elements": [{"type": "relation", "id": 1}]}

        return Resp()

    monkeypatch.setattr("core.utils.waterbody.requests.post", fake_post)
    monkeypatch.setattr(
        "core.utils.waterbody._is_relation_bbox_too_large", lambda rel_id, max_deg=2.0: False
    )
    monkeypatch.setattr(
        "core.utils.waterbody.fetch_waterbody_polygon_osmnx",
        lambda rel_id: Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
    )
    poly = fetch_waterbody_polygon(0.5, 0.5, radius_km=1)
    assert isinstance(poly, Polygon)
    assert poly.contains(Point(0.5, 0.5))


def test_fetch_waterbody_polygon_none(monkeypatch):
    def fake_post(url, data=None, timeout=30):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"elements": []}

        return Resp()

    monkeypatch.setattr("core.utils.waterbody.requests.post", fake_post)
    assert fetch_waterbody_polygon(0, 0, radius_km=1) is None


def test_generate_contours_with_water_polygon(tmp_path):
    from core.utils import slicer
    slicer.DEBUG_IMAGE_PATH = str(tmp_path)
    elevation = np.array(
        [[0, 100, 200], [0, 100, 200], [0, 100, 200]], dtype=float
    )
    transform = rasterio.transform.from_origin(0, 3, 1, 1)
    water_poly = Polygon([(1.1, 1.9), (1.9, 1.9), (1.9, 1.1), (1.1, 1.1)])
    layers = generate_contours(
        elevation,
        transform,
        interval=100,
        num_layers=2,
        fixed_elevation=100,
        water_polygon=water_poly,
        debug_image_path=str(tmp_path),
    )
    elevations = [l["elevation"] for l in layers]
    # One band is inserted at the fixed elevation (water band) and
    # additional edges are added slightly below and above. Only a single
    # layer should exactly match the fixed elevation.
    assert elevations.count(100.0) == 1


def test_water_polygon_clipped_to_bounds(tmp_path):
    from core.utils import slicer
    slicer.DEBUG_IMAGE_PATH = str(tmp_path)
    elev = np.array([[0, 0], [100, 100]], dtype=float)
    transform = rasterio.transform.from_origin(0, 2, 1, 1)
    big_water = Polygon([(-0.5, 2.5), (2.5, 2.5), (2.5, -0.5), (-0.5, -0.5)])
    layers = generate_contours(
        elev,
        transform,
        interval=100,
        fixed_elevation=50,
        water_polygon=big_water,
        debug_image_path=str(tmp_path),
    )
    from shapely.geometry import shape
    water_geom = shape(next(l["geometry"] for l in layers if l["elevation"] == 50.0))
    # The water polygon is inserted unchanged; it is not clipped to the
    # elevation bounds.
    assert water_geom.equals(big_water)

