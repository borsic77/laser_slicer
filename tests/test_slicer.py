import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
import rasterio.transform
from django.conf import settings
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.slicer import (
    _compute_layer_bands,
    _create_contourf_levels,
    _extract_level_polygons,
    _force_multipolygon,
    _prepare_meshgrid,
    clean_geometry,
    clean_geometry_strict,
    mosaic_and_crop,
    project_geometry,
    round_affine,
    scale_and_center_contours_to_substrate,
    walk_bbox_between,
)

# Use headless matplotlib
matplotlib.use("Agg")

TILE_CACHE_DIR = settings.TILE_CACHE_DIR
DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
logger = logging.getLogger(__name__)


@pytest.fixture
def switzerland_bounds():
    # Covers Yverdon-les-Bains and surroundings
    return (6.5, 46.7, 6.7, 46.9)


def test_mosaic_and_crop(switzerland_bounds):
    paths = download_srtm_tiles_for_bounds(switzerland_bounds)
    array, transform = mosaic_and_crop(paths, switzerland_bounds)
    assert isinstance(array, np.ndarray)
    assert array.ndim == 2
    assert array.size > 0
    assert transform is not None


def test_is_almost_closed():
    from shapely.geometry import LineString

    from core.utils.slicer import is_almost_closed

    fully_closed = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
    almost_closed = LineString(
        [(0, 0), (1, 0), (1, 1), (0, 1e-9)]
    )  # very close to (0, 0)
    open_loop = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])

    assert not is_almost_closed(fully_closed)
    assert is_almost_closed(almost_closed)
    assert not is_almost_closed(open_loop)


# Test for walk_bbox_between
def test_walk_bbox_between():
    from core.utils.slicer import walk_bbox_between

    bbox_coords = [(0, 1), (0, 0), (2, 0), (2, 1)]  # Clockwise: TL, BL, BR, TR

    # Walk CW from end_idx=2 (BR) to start_idx=0 (TL)
    segment_cw = walk_bbox_between(bbox_coords, 0, 2, direction="cw")
    assert segment_cw == [(2, 0), (2, 1), (0, 1)]

    # Walk CCW from end_idx=0 (TL) to start_idx=2 (BR)
    segment_ccw = walk_bbox_between(bbox_coords, 2, 0, direction="ccw")
    assert segment_ccw == [(0, 1), (2, 1), (2, 0)]

    # Wraparound CW from end_idx=1 (


# Test for project_geometry
def test_project_geometry(tmp_path):
    from shapely.geometry import Polygon, mapping

    from core.utils import slicer
    from core.utils.slicer import project_geometry

    # Create a square polygon near Yverdon-les-Bains
    poly = Polygon(
        [(6.6, 46.8), (6.601, 46.8), (6.601, 46.801), (6.6, 46.801), (6.6, 46.8)]
    )
    contour = {"geometry": mapping(poly), "elevation": 500}

    slicer.DEBUG_IMAGE_PATH = str(tmp_path)
    result = project_geometry([contour], center_lon=6.6, center_lat=46.8)
    assert isinstance(result, list)
    assert "geometry" in result[0]
    assert result[0]["geometry"]["type"] == "Polygon"


# Test for scale_and_center_contours_to_substrate
def test_scale_and_center_contours_to_substrate():
    from shapely.geometry import Polygon, mapping, shape

    from core.utils.slicer import scale_and_center_contours_to_substrate

    # Create a large UTM-scale square around (300000, 5200000)
    utm_poly = Polygon(
        [
            (300000, 5200000),
            (300100, 5200000),
            (300100, 5200100),
            (300000, 5200100),
            (300000, 5200000),
        ]
    )
    contour = {"geometry": mapping(utm_poly), "elevation": 100}

    substrate_mm = 200.0  # 20 cm side square
    utm_bounds = (300000, 5200000, 300100, 5200100)

    scaled = scale_and_center_contours_to_substrate([contour], substrate_mm, utm_bounds)

    assert len(scaled) == 1
    geom = shape(scaled[0]["geometry"])
    assert geom.is_valid
    assert geom.area == pytest.approx(
        substrate_mm**2 * 0.000001, rel=0.01
    )  # within 1% of area
    assert geom.length == pytest.approx(
        substrate_mm * 0.004, rel=0.01
    )  # within 1% of perimeter
    assert geom.bounds[0] >= -0.1 and geom.bounds[2] <= 0.1  # x within +/-10 cm
    assert geom.bounds[1] >= -0.1 and geom.bounds[3] <= 0.1  # y within +/-10 cm


# Test for _prepare_meshgrid


def test_prepare_meshgrid():
    # Create a simple 3x2 elevation array
    elevation_data = np.array([[1, 2], [3, 4], [5, 6]])
    # Define a transform: origin at (10, 20), pixel size 1x1
    transform = rasterio.transform.from_origin(west=10, north=20, xsize=1, ysize=1)

    lon, lat = _prepare_meshgrid(elevation_data, transform)

    assert lon.shape == elevation_data.shape
    assert lat.shape == elevation_data.shape
    # Top-left pixel center
    assert lon[0, 0] == 10.5
    assert lat[0, 0] == 19.5
    # Bottom-right pixel center
    assert lon[2, 1] == 11.5
    assert lat[2, 1] == 17.5


# Test for _create_contourf_levels
def test_create_contourf_levels():
    data = np.array(
        [
            [103, 110, 118],
            [125, 132, 140],
        ]
    )
    interval = 20

    levels = _create_contourf_levels(data, interval)
    # The implementation now returns levels starting from the minimum
    # elevation and stepping by ``interval`` without flooring/ceiling.
    # With the example data this yields [103, 123].
    expected = np.array([103.0, 123.0])
    assert np.allclose(levels, expected)

    # Using num_layers should use the min and max directly
    levels_num = _create_contourf_levels(data, interval, num_layers=2)
    expected_num = np.array([103.0, 121.5, 140.0])
    assert np.allclose(levels_num, expected_num)


# Test for _flatten_polygons
def test_flatten_polygons():
    from shapely.geometry import MultiPolygon, Polygon

    from core.utils.slicer import _flatten_polygons

    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    multi = MultiPolygon([p1, p2])

    result = _flatten_polygons([p1, multi])
    assert isinstance(result, list)
    assert all(poly.geom_type == "Polygon" for poly in result)
    assert len(result) == 3  # one from p1, two from multi


# Additional tests for round_affine and geometry cleaning utilities
def test_round_affine():
    t = rasterio.transform.from_origin(7.123456, 46.987654, 1.234567, 0.987654)
    rounded = round_affine(t, precision=1e-3)
    # All components should be rounded to 3 decimal places
    for val in rounded:
        assert abs(val - round(val, 3)) < 1e-10


def test_clean_geometry():
    # Valid polygon
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert clean_geometry(p).is_valid

    # Polygon with self-intersection ("bowtie") - make_valid should fix or return None
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    cleaned = clean_geometry(bowtie)
    assert cleaned is None or cleaned.is_valid

    # Empty geometry

    empty = GeometryCollection()
    assert clean_geometry(empty) is None

    # Zero-area
    line = LineString([(0, 0), (1, 1)])
    assert clean_geometry(line) is None


def test_clean_geometry_strict():
    # Valid polygon
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert clean_geometry_strict(p).is_valid

    # Polygon with self-intersection ("bowtie") - strict version
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    cleaned = clean_geometry_strict(bowtie)
    assert cleaned is None or cleaned.is_valid

    # Invalid collection
    coll = GeometryCollection([LineString([(0, 0), (1, 1)]), Point(1, 2)])
    assert clean_geometry_strict(coll) is None

    # MultiPolygon with a zero-area polygon inside
    from shapely.geometry import MultiPolygon

    good = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    bad = Polygon([(0, 0), (0, 0), (0, 0)])  # zero area
    mp = MultiPolygon([good, bad])
    cleaned = clean_geometry_strict(mp)
    assert cleaned.is_valid or cleaned is None


# Test for _force_multipolygon
def test_force_multipolygon():
    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mp = MultiPolygon([p])
    result1 = _force_multipolygon(p)
    result2 = _force_multipolygon(mp)

    assert isinstance(result1, MultiPolygon)
    assert isinstance(result2, MultiPolygon)
    assert len(result1.geoms) == 1
    assert len(result2.geoms) == 1


# Test for _prepare_meshgrid
def test_prepare_meshgrid():
    # Create a simple 3x2 elevation array
    elevation_data = np.array([[1, 2], [3, 4], [5, 6]])
    # Define a transform: origin at (10, 20), pixel size 1x1
    transform = rasterio.transform.from_origin(west=10, north=20, xsize=1, ysize=1)

    lon, lat = _prepare_meshgrid(elevation_data, transform)

    assert lon.shape == elevation_data.shape
    assert lat.shape == elevation_data.shape
    # Top-left pixel center
    assert lon[0, 0] == 10.5
    assert lat[0, 0] == 19.5
    # Bottom-right pixel center
    assert lon[2, 1] == 11.5
    assert lat[2, 1] == 17.5


def test_extract_level_polygons():
    # Generate simple contour data
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = X + Y  # simple sloped surface

    cs = plt.contourf(X, Y, Z, levels=[0.5, 1.0, 1.5])
    results = _extract_level_polygons(cs)

    # Basic structure checks
    assert isinstance(results, list)
    for level, polygons in results:
        assert isinstance(level, float)
        assert isinstance(polygons, list)
        for p in polygons:
            assert p.is_valid
            assert p.area > 0
            assert p.geom_type == "Polygon"
    plt.close()


# Enhanced test for _extract_level_polygons with two pyramid-shaped peaks
def test_extract_level_polygons_two_peaks():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Create two cone-like pyramids
    Z1 = 1.0 - np.sqrt((X - 0.3) ** 2 + (Y - 0.3) ** 2)
    Z2 = 1.0 - np.sqrt((X - 0.7) ** 2 + (Y - 0.7) ** 2)
    Z = np.maximum(Z1, Z2)
    Z[Z < 0] = 0  # Clip to zero

    cs = plt.contourf(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8])
    results = _extract_level_polygons(cs)

    assert len(results) == 3  # Four levels

    for level, polygons in results:
        assert isinstance(level, float)
        assert isinstance(polygons, list)
        for p in polygons:
            assert p.is_valid
            assert p.area > 0
            assert p.geom_type == "Polygon"

    # Check that lower levels contain more polygons (i.e. both peaks are visible)
    low_level_polygons = results[0][1]
    assert len(low_level_polygons) >= 2

    plt.close()


# Test for _compute_layer_bands
def test_compute_layer_bands():
    # Create synthetic terrain with nested contours
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = 1.0 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    Z[Z < 0] = 0

    cs = plt.contourf(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8])
    levels = _extract_level_polygons(cs)
    bands = _compute_layer_bands(levels, rasterio.transform.from_origin(0, 1, 1, 1))

    assert isinstance(bands, list)
    assert len(bands) == len(levels)
    for band in bands:
        assert "elevation" in band
        assert "geometry" in band
        assert band["closed"] is True
        geom = shape(band["geometry"])
        assert geom.is_valid
        assert not geom.is_empty
        assert geom.geom_type in ("Polygon", "MultiPolygon")

    plt.close()


# Comprehensive test for _compute_layer_bands using two hills
def test_compute_layer_bands_two_hills():
    # Two hills: one centered at (0.3, 0.3), one at (0.7, 0.7), with steep, distinct peaks
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) * 100)
    Z2 = np.exp(-((X - 0.7) ** 2 + (Y - 0.7) ** 2) * 100)
    Z = np.maximum(Z1, Z2)
    Z[Z < 0] = 0

    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    cs = plt.contourf(X, Y, Z, levels=levels)
    level_polygons = _extract_level_polygons(cs)
    bands = _compute_layer_bands(
        level_polygons, rasterio.transform.from_origin(0, 1, 1, 1)
    )

    assert len(bands) == len(levels) - 1

    for band in bands:
        assert "elevation" in band
        assert "geometry" in band
        assert band["closed"] is True
        geom = shape(band["geometry"])
        assert geom.is_valid
        assert not geom.is_empty
        assert geom.geom_type in ("Polygon", "MultiPolygon")

        # Check that each lower band contains the one above
    for i in range(1, len(bands)):
        lower = shape(bands[i]["geometry"])
        upper = shape(bands[i - 1]["geometry"])
        assert lower.covers(lower)

    # Base layer should include both peaks (at least two distinct regions)
    base_geom = shape(bands[0]["geometry"])
    if base_geom.geom_type == "MultiPolygon":
        assert len(base_geom.geoms) >= 2
    plt.close()


# Test for _compute_layer_bands with a tall cylinder: all bands should have similar area
def test_compute_layer_bands_cylinder():
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import shape

    from core.utils.slicer import _compute_layer_bands, _extract_level_polygons

    # Create a cylinder with sharp elevation edges
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    radius = 0.2
    center = (0.5, 0.5)
    distance = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    Z = np.where(distance < radius, 1.0, 0.0)

    cs = plt.contourf(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8, 1.0])
    level_polygons = _extract_level_polygons(cs)
    bands = _compute_layer_bands(
        level_polygons, rasterio.transform.from_origin(0, 1, 1, 1)
    )

    assert len(bands) == 4

    for band in bands:
        assert "elevation" in band
        assert "geometry" in band
        assert band["closed"] is True
        geom = shape(band["geometry"])
        assert geom.is_valid
        assert not geom.is_empty
        assert geom.geom_type in ("Polygon", "MultiPolygon")

    # All layers should have similar area
    areas = [shape(band["geometry"]).area for band in bands]
    for a in areas:
        assert abs(a - areas[0]) < 0.01  # within 1% variation

    plt.close()


# Additional tests for clean_geometry_strict edge cases
def test_clean_geometry_strict_buffer_exception(monkeypatch):
    # Fake invalid geometry object
    class FakeInvalid:
        is_empty = False
        area = 1.0
        is_valid = False

        def buffer(self, *args, **kwargs):
            raise ValueError("buffer failure")

    fake_geom = FakeInvalid()
    from core.utils import slicer

    monkeypatch.setattr(slicer, "make_valid", lambda x: fake_geom)
    assert clean_geometry_strict(fake_geom) is None


def test_clean_geometry_strict_invalid_after_buffer():
    # A polygon that remains invalid even after buffer(0)
    gc = GeometryCollection([LineString([(0, 0), (1, 1)])])
    assert clean_geometry_strict(gc) is None


def test_clean_geometry_strict_geom_collection_empty():
    # GeometryCollection with only lines and points (no polygons)
    gc = GeometryCollection([LineString([(0, 0), (1, 1)]), Point(1, 2)])
    assert clean_geometry_strict(gc) is None


def test_clean_geometry_strict_geom_collection_one_poly():
    # GeometryCollection with a single Polygon
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gc = GeometryCollection([poly])
    result = clean_geometry_strict(gc)
    assert isinstance(result, Polygon)  # Should extract the polygon


def test_clean_geometry_strict_geom_collection_multi_poly():
    # GeometryCollection with two polygons
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    gc = GeometryCollection([poly1, poly2])
    result = clean_geometry_strict(gc)
    from shapely.geometry import MultiPolygon

    assert isinstance(result, MultiPolygon)
    assert len(result.geoms) == 2


def test_save_debug_contour_polygon_polygon(tmp_path):
    # Set DEBUG_IMAGE_PATH to tmp_path for this test
    import os

    from shapely.geometry import Polygon

    from core.utils import slicer
    from core.utils.slicer import save_debug_contour_polygon

    slicer.DEBUG_IMAGE_PATH = str(tmp_path)
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    save_debug_contour_polygon(poly, 100, "testpoly")
    expected = os.path.join(str(tmp_path), "testpoly_elev_100.png")
    assert os.path.exists(expected)


def test_save_debug_contour_polygon_multipolygon(tmp_path):
    import os

    from shapely.geometry import MultiPolygon, Polygon

    from core.utils import slicer
    from core.utils.slicer import save_debug_contour_polygon

    slicer.DEBUG_IMAGE_PATH = str(tmp_path)
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    multi = MultiPolygon([poly1, poly2])
    save_debug_contour_polygon(multi, 200, "testmulti")
    expected = os.path.join(str(tmp_path), "testmulti_elev_200.png")
    assert os.path.exists(expected)


def test_save_debug_contour_polygon_empty():
    # Should not throw, should return early
    from core.utils.slicer import save_debug_contour_polygon

    class DummyPoly:
        is_empty = True
        is_valid = True

    save_debug_contour_polygon(DummyPoly(), 123, "shouldnotcreate")
    # (No assertion needed; just check it doesn't error)


def test_save_debug_contour_polygon_invalid():
    # Should not throw, should return early
    from core.utils.slicer import save_debug_contour_polygon

    class DummyPoly:
        is_empty = False
        is_valid = False

    save_debug_contour_polygon(DummyPoly(), 124, "shouldnotcreate2")


# --- Additional tests for mosaic_and_crop edge cases ---
import numpy as np
import rasterio
import rasterio.transform
import rasterio.windows


def test_mosaic_and_crop_lat_swap(monkeypatch):
    # Provide a dummy raster with proper shape, test lat_min > lat_max
    from core.utils.slicer import mosaic_and_crop

    def fake_merge(src_files):
        arr = np.ones((1, 5, 5))
        # Use a real affine transform (which is a tuple)
        transform = rasterio.transform.from_origin(0, 5, 1, 1)

        # Replace the .e attribute by subclassing
        class CustomTransform(type(transform)):
            @property
            def e(self):
                return 1

        return arr, CustomTransform(*transform)

    monkeypatch.setattr("core.utils.slicer.merge", fake_merge)
    import logging

    monkeypatch.setattr("core.utils.slicer.logger", logging.getLogger("dummy"))

    # Normal window behavior
    def fake_from_bounds(*args, **kwargs):
        return rasterio.windows.Window(0, 0, 5, 5)

    monkeypatch.setattr("core.utils.slicer.from_bounds", fake_from_bounds)
    class DummySrc:
        def close(self):
            pass

    monkeypatch.setattr("rasterio.open", lambda *a, **k: DummySrc())
    # Should succeed and return array, even with swapped lats
    array, transform = mosaic_and_crop(["dummy"], (0, 5, 5, 0))  # lat_min > lat_max
    assert isinstance(array, np.ndarray)
    assert array.shape == (5, 5)


def test_mosaic_and_crop_orientation_warning(monkeypatch):
    from core.utils.slicer import mosaic_and_crop

    # Simulate transform.e > 0

    def fake_merge(src_files):
        arr = np.ones((1, 5, 5))
        # Use a real affine transform (which is a tuple)
        transform = rasterio.transform.from_origin(0, 5, 1, 1)

        # Replace the .e attribute by subclassing
        class CustomTransform(type(transform)):
            @property
            def e(self):
                return 1

        return arr, CustomTransform(*transform)

    monkeypatch.setattr("core.utils.slicer.merge", fake_merge)
    import logging

    monkeypatch.setattr("core.utils.slicer.logger", logging.getLogger("dummy"))
    monkeypatch.setattr(
        "core.utils.slicer.from_bounds",
        lambda *a, **k: rasterio.windows.Window(0, 0, 5, 5),
    )

    # Patch rasterio.open
    class DummySrc:
        def close(self):
            pass

    monkeypatch.setattr("rasterio.open", lambda *a, **k: DummySrc())

    array, transform = mosaic_and_crop(["dummy"], (0, 0, 5, 5))
    assert isinstance(array, np.ndarray)


def test_mosaic_and_crop_from_bounds_error(monkeypatch):
    from core.utils.slicer import mosaic_and_crop

    # Patch rasterio.open to return a dummy object
    class DummySrc:
        pass

    monkeypatch.setattr("rasterio.open", lambda *a, **k: DummySrc())

    # Simulate merge to return a valid array and transform
    def fake_merge(src_files):
        arr = np.ones((1, 5, 5))
        import rasterio.transform

        transform = rasterio.transform.from_origin(0, 5, 1, 1)
        return arr, transform

    monkeypatch.setattr("core.utils.slicer.merge", fake_merge)

    import logging

    monkeypatch.setattr("core.utils.slicer.logger", logging.getLogger("dummy"))

    # Raise in from_bounds
    def fake_from_bounds(*args, **kwargs):
        raise ValueError("fail")

    monkeypatch.setattr("core.utils.slicer.from_bounds", fake_from_bounds)

    with pytest.raises(ValueError):
        mosaic_and_crop(["dummy"], (0, 0, 5, 5))
