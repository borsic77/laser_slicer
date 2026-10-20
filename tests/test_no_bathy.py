import logging

import numpy as np
import rasterio.transform

from core.services.contour_generator import ContourSlicingJob

logger = logging.getLogger(__name__)


def test_no_bathymetry_clamping(monkeypatch):
    """Ensure coastal clamping forces ocean voids and negative noise to non-negative contours when bathymetry is disabled."""
    lat = 28.11924
    lon = -17.22575
    delta = 0.01
    bounds = (lon - delta, lat - delta, lon + delta, lat + delta)
    center = (lon, lat)

    # Synthetic DEM with ocean voids (-32768) and negative values on one half,
    # and land elevation (50 to 300) on the other.
    dem_rows = []
    for r in range(10):
        row = []
        for c in range(10):
            if c < 4:
                row.append(-32768.0 if c < 2 else -50.0)
            else:
                row.append(50.0 + (c - 4) * 50.0 + r * 5.0)
        dem_rows.append(row)
    dem = np.array(dem_rows, dtype=np.float32)

    transform = rasterio.transform.from_bounds(*bounds, 10, 10)

    monkeypatch.setattr(
        "core.utils.dem.download_elevation_tiles_for_bounds",
        lambda b: ["mock_tile.tif"],
    )
    monkeypatch.setattr(
        "core.utils.dem.mosaic_and_crop",
        lambda paths, b, **k: (dem.copy(), transform),
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

    # 1. Run without bathymetry
    job = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=50.0,
        num_layers=5,
        simplify=0.5,
        substrate_size_mm=400,
        layer_thickness_mm=5,
        center=center,
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_bathymetry=False,
        use_osm_water_mask=False,
    )

    contours = job.run()
    elevations = [c["elevation"] for c in contours]
    assert len(contours) > 0
    assert all(elev >= -1.0 for elev in elevations), (
        f"Expected all elevations >= -1.0, got {elevations}"
    )
    assert not any(elev < -1.0 for elev in elevations), (
        f"Expected no elevations < -1.0, got {elevations}"
    )

    # 2. Run with bathymetry enabled
    bathy_dem = np.full((10, 10), -200.0, dtype=np.float32)
    monkeypatch.setattr(
        "core.services.bathymetry_service.BathymetryFetcher.fetch_elevation_for_bounds",
        lambda self, b: bathy_dem,
    )

    job_bathy = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=50.0,
        num_layers=5,
        simplify=0.5,
        substrate_size_mm=400,
        layer_thickness_mm=5,
        center=center,
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_bathymetry=True,
        use_osm_water_mask=False,
    )

    contours_bathy = job_bathy.run()
    elevations_bathy = [c["elevation"] for c in contours_bathy]
    assert len(contours_bathy) > 0
    assert any(elev < 0 for elev in elevations_bathy), (
        f"Expected negative elevations with bathymetry, got {elevations_bathy}"
    )
