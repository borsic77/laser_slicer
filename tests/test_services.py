import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio.transform
import shapely.geometry
from shapely.geometry import Polygon, mapping

# Provide dummy 'django.conf' with a settings object used by contour_generator
django_conf = types.ModuleType("django.conf")
django_conf.settings = types.SimpleNamespace(
    DEBUG_IMAGE_PATH=None,
    TILE_CACHE_DIR=Path("/tmp"),
    DEBUG=False,
    MEDIA_ROOT="/tmp",
    NOMINATIM_USER_AGENT="test-agent",
)
sys.modules.setdefault("django.conf", django_conf)
cache_module = types.ModuleType("django.core.cache")
cache_module.cache = types.SimpleNamespace(
    set=lambda *a, **k: None, get=lambda *a, **k: None
)
sys.modules.setdefault("django.core.cache", cache_module)

from core.services.contour_generator import ContourSlicingJob
from core.services.elevation_service import ElevationDataError, ElevationRangeJob


def test_contour_slicing_job_run(monkeypatch):
    """Ensure ContourSlicingJob.run integrates utilities correctly."""
    recorded = {}

    def fake_download(bounds):
        recorded["download"] = bounds
        return ["tile1"]

    def fake_mosaic(paths, bounds, **kwargs):
        recorded["mosaic"] = (paths, bounds)
        arr = np.ones((2, 2), dtype=float)
        transform = rasterio.transform.from_origin(0, 2, 1, 1)
        return arr, transform

    fake_contours = [
        {
            "elevation": 100.0,
            "geometry": mapping(Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])),
            "closed": True,
        }
    ]

    def fake_generate(
        masked_elevation_data, elevation_data, transform, interval, simplify, **kwargs
    ):
        recorded["generate"] = {
            "height": interval,
            "simplify": simplify,
            "center": kwargs.get("center"),
            "bounds": kwargs.get("bounds"),
            "fixed_elevation": kwargs.get("fixed_elevation"),
            "num_layers": kwargs.get("num_layers"),
        }
        return fake_contours

    monkeypatch.setattr(
        "core.utils.dem.download_elevation_tiles_for_bounds",
        fake_download,
    )
    monkeypatch.setattr("core.utils.dem.mosaic_and_crop", fake_mosaic)
    monkeypatch.setattr("core.utils.dem.clean_srtm_dem", lambda x, **k: x)
    # robust_local_outlier_mask is commented out in code, so no need to mock it

    monkeypatch.setattr("core.utils.contour_ops.generate_contours", fake_generate)
    monkeypatch.setattr(
        "core.utils.geometry_ops.project_geometry",
        lambda c, cx, cy, simplify_tolerance=0, existing_transform=None: (
            c,
            ("proj", (0, 0), 0),
        ),
    )
    # Obsolete patches removed for smooth_geometry, clip_contours, scale, filter_small_features
    monkeypatch.setattr(
        "core.utils.geocoding.compute_utm_bounds_from_wgs84",
        lambda *a: (0, 0, 1, 1),
    )
    monkeypatch.setattr(
        "core.services.contour_generator._log_contour_info", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "core.utils.parallel_ops.process_and_scale_single_contour",
        lambda c, *args: c,
    )

    class FakePool:
        def __init__(self, processes=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def starmap(self, func, iterable):
            # func is the mock lambda c, *args: c
            # iterable is a list of tuples (chunks of arguments)
            return [func(*args) for args in iterable]

    monkeypatch.setattr("billiard.Pool", FakePool)

    job = ContourSlicingJob(
        bounds=(0.0, 0.0, 1.0, 1.0),
        height_per_layer=100.0,
        num_layers=1,
        simplify=0.0,
        substrate_size_mm=100.0,
        layer_thickness_mm=2.0,
        center=(0.5, 0.5),
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
    )

    result = job.run()
    assert recorded["download"] == job.bounds
    assert recorded["mosaic"] == (["tile1"], job.bounds)
    assert recorded["generate"]["height"] == 100.0
    assert recorded["generate"]["center"] == job.center
    assert isinstance(result, list) and len(result) == 1
    assert result[0]["elevation"] == 100.0
    assert "thickness" in result[0]
    assert result[0]["thickness"] == pytest.approx(0.002)


def test_contour_slicing_job_with_osm(monkeypatch):
    poly = Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)])
    ml = shapely.geometry.MultiLineString([[(0, 0.75), (1, 0.75)]])

    monkeypatch.setattr(
        "core.utils.dem.download_elevation_tiles_for_bounds",
        lambda b: ["tile"],
    )
    monkeypatch.setattr(
        "core.utils.dem.mosaic_and_crop",
        lambda p, b, **k: (np.ones((1, 1)), None),
    )
    monkeypatch.setattr("core.utils.dem.clean_srtm_dem", lambda x, **k: x)
    monkeypatch.setattr(
        "core.utils.contour_ops.generate_contours",
        lambda *a, **k: [{"elevation": 0, "geometry": mapping(poly), "closed": True}],
    )
    monkeypatch.setattr(
        "core.utils.geometry_ops.project_geometry",
        lambda c, cx, cy, simplify_tolerance=0, existing_transform=None: (
            c,
            ("proj", (0, 0), 0),
        ),
    )
    # Obsolete patches removed
    monkeypatch.setattr(
        "core.utils.geocoding.compute_utm_bounds_from_wgs84",
        lambda *a: (0, 0, 1, 1),
    )
    monkeypatch.setattr(
        "core.services.contour_generator._log_contour_info", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "core.utils.parallel_ops.process_and_scale_single_contour",
        lambda c, *args: c,
    )
    monkeypatch.setattr(
        "core.utils.osm_features.fetch_roads",
        lambda b: {"residential": ml},
    )
    monkeypatch.setattr("core.utils.osm_features.fetch_waterways", lambda b: ml)
    monkeypatch.setattr(
        "core.utils.osm_features.fetch_buildings",
        lambda b: shapely.geometry.MultiPolygon([poly]),
    )

    class FakePool:
        def __init__(self, processes=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def starmap(self, func, iterable):
            return [func(*args) for args in iterable]

    monkeypatch.setattr("billiard.Pool", FakePool)

    job = ContourSlicingJob(
        bounds=(0, 0, 1, 1),
        height_per_layer=100,
        num_layers=1,
        simplify=0,
        substrate_size_mm=1000,
        layer_thickness_mm=2,
        center=(0.5, 0.5),
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_roads=True,
        include_buildings=True,
        include_waterways=True,
    )
    result = job.run()
    layer = result[0]
    assert "roads" in layer and isinstance(layer["roads"], dict)
    assert "residential" in layer["roads"]
    assert "buildings" in layer
    assert "waterways" in layer


def test_contour_slicing_job_empty_osm(monkeypatch):
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])

    monkeypatch.setattr(
        "core.utils.dem.download_elevation_tiles_for_bounds",
        lambda b: ["tile"],
    )
    monkeypatch.setattr(
        "core.utils.dem.mosaic_and_crop",
        lambda p, b, **k: (np.ones((1, 1)), None),
    )
    monkeypatch.setattr("core.utils.dem.clean_srtm_dem", lambda x, **k: x)
    monkeypatch.setattr(
        "core.utils.contour_ops.generate_contours",
        lambda *a, **k: [{"elevation": 0, "geometry": mapping(poly), "closed": True}],
    )
    monkeypatch.setattr(
        "core.utils.geometry_ops.project_geometry",
        lambda c, cx, cy, simplify_tolerance=0, existing_transform=None: (
            c,
            ("proj", (0, 0), 0),
        ),
    )
    # Obsolete patches removed
    monkeypatch.setattr(
        "core.utils.geocoding.compute_utm_bounds_from_wgs84",
        lambda *a: (0, 0, 1, 1),
    )
    monkeypatch.setattr(
        "core.services.contour_generator._log_contour_info", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "core.utils.parallel_ops.process_and_scale_single_contour",
        lambda c, *args: c,
    )
    monkeypatch.setattr(
        "core.utils.osm_features.fetch_roads",
        lambda b: {},
    )
    monkeypatch.setattr(
        "core.utils.osm_features.fetch_waterways",
        lambda b: shapely.geometry.MultiLineString(),
    )
    monkeypatch.setattr(
        "core.utils.osm_features.fetch_buildings",
        lambda b: shapely.geometry.MultiPolygon(),
    )

    class FakePool:
        def __init__(self, processes=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def starmap(self, func, iterable):
            return [func(*args) for args in iterable]

    monkeypatch.setattr("billiard.Pool", FakePool)

    job = ContourSlicingJob(
        bounds=(0, 0, 1, 1),
        height_per_layer=100,
        num_layers=1,
        simplify=0,
        substrate_size_mm=100,
        layer_thickness_mm=2,
        center=(0.5, 0.5),
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_roads=True,
        include_buildings=True,
        include_waterways=True,
    )
    result = job.run()
    layer = result[0]
    assert layer.get("roads") is None
    assert layer.get("waterways") is None
    assert layer.get("buildings") is None


def test_elevation_range_job(monkeypatch):
    """ElevationRangeJob.run returns min and max elevations."""
    arr = np.array([[100, 200], [300, 400]], dtype=float)
    monkeypatch.setattr(
        "core.utils.dem.download_elevation_tiles_for_bounds",
        lambda b: ["tile"],
    )
    monkeypatch.setattr("core.utils.dem.mosaic_and_crop", lambda p, b, **k: (arr, None))
    monkeypatch.setattr("core.utils.dem.clean_srtm_dem", lambda x, **k: x)

    job = ElevationRangeJob((0, 0, 1, 1))
    result = job.run()
    assert result == {"min": 100.0, "max": 400.0}


def test_elevation_range_job_invalid(monkeypatch):
    """ElevationRangeJob.run raises ElevationDataError when DEM is invalid."""
    arr = np.array([[np.nan, np.nan]])
    monkeypatch.setattr(
        "core.utils.dem.download_elevation_tiles_for_bounds",
        lambda b: ["tile"],
    )
    monkeypatch.setattr("core.utils.dem.mosaic_and_crop", lambda p, b, **k: (arr, None))
    monkeypatch.setattr("core.utils.dem.clean_srtm_dem", lambda x, **k: x)

    job = ElevationRangeJob((0, 0, 1, 1))
    with pytest.raises(ElevationDataError):
        job.run()
