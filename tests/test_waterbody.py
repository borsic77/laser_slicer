import numpy as np
import pytest
from shapely.geometry import Point, Polygon

import rasterio.transform
from core.utils.waterbody import fetch_waterbody_polygon
from core.utils.slicer import generate_contours


def test_fetch_waterbody_polygon_found(monkeypatch):
    def fake_post(url, data=None, timeout=30):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "elements": [
                        {
                            "type": "way",
                            "geometry": [
                                {"lat": 0, "lon": 0},
                                {"lat": 0, "lon": 1},
                                {"lat": 1, "lon": 1},
                                {"lat": 1, "lon": 0},
                                {"lat": 0, "lon": 0},
                            ],
                        }
                    ]
                }

        return Resp()

    monkeypatch.setattr("core.utils.waterbody.requests.post", fake_post)
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
    assert elevations.count(100.0) == 2


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
    from shapely.geometry import box, shape
    bbox = box(0, 0, 2, 2)
    water_geom = shape(next(l["geometry"] for l in layers if l["elevation"] == 50.0))
    assert water_geom.equals(big_water.intersection(bbox))

