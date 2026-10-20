import io

import numpy as np
import rasterio
import rasterio.transform

from core.services.bathymetry_service import BathymetryFetcher
from core.utils.dem import (
    get_great_lakes_surface_elevation,
    is_in_great_lakes,
    merge_srtm_and_etopo,
)


def _create_geotiff_bytes(
    data: np.ndarray, bounds: tuple[float, float, float, float]
) -> bytes:
    mem = io.BytesIO()
    height, width = data.shape
    transform = rasterio.transform.from_bounds(*bounds, width, height)
    with rasterio.open(
        mem,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return mem.getvalue()


def test_is_in_europe():
    fetcher = BathymetryFetcher()
    # European bounds: Macaronesia (La Gomera), Mediterranean, North Sea
    assert fetcher.is_in_europe((-17.340, 28.090, -17.330, 28.100)) is True
    assert fetcher.is_in_europe((8.5, 47.3, 8.6, 47.4)) is True

    # Global bounds outside Europe: Mariana Trench, Americas, Pacific
    assert fetcher.is_in_europe((142.15, 11.30, 142.25, 11.40)) is False
    assert fetcher.is_in_europe((-122.5, 37.7, -122.4, 37.8)) is False
    assert fetcher.is_in_europe((-60.0, 10.0, -50.0, 20.0)) is False


def test_emodnet_fetch(tmp_path, monkeypatch):
    """Test fetching European bathymetry using mocked EMODnet WCS."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    fetcher = BathymetryFetcher(cache_dir=tmp_path)
    bounds_eu = (-17.340, 28.090, -17.330, 28.100)
    assert fetcher.is_in_europe(bounds_eu) is True

    synth_data = np.full((10, 10), -250.0, dtype=np.float32)
    tif_bytes = _create_geotiff_bytes(synth_data, bounds_eu)

    get_called = 0

    class FakeResponse:
        def __init__(self, content):
            self.content = content
            self.status_code = 200

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=30):
        nonlocal get_called
        get_called += 1
        return FakeResponse(tif_bytes)

    monkeypatch.setattr("core.services.bathymetry_service.requests.get", fake_get)

    data = fetcher.fetch_elevation_for_bounds(bounds_eu)
    assert get_called == 1
    assert data.shape == (10, 10)
    assert np.nanmin(data) == -250.0

    # Ensure caching works on repeated request without extra HTTP call
    data_cached = fetcher.fetch_elevation_for_bounds(bounds_eu)
    assert get_called == 1
    np.testing.assert_array_equal(data, data_cached)


def test_gebco_fetch_global(tmp_path, monkeypatch):
    """Test fetching global bathymetry using mocked GEBCO rasterio dataset."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetcher = BathymetryFetcher(cache_dir=cache_dir)
    bounds_global = (142.15, 11.30, 142.25, 11.40)
    assert fetcher.is_in_europe(bounds_global) is False

    gebco_file = tmp_path / "gebco_mock.tif"
    width, height = 30, 30
    synth_gebco = np.full((height, width), -6500.0, dtype=np.float32)
    synth_gebco[0, 0] = -32768.0  # nodata handling
    transform = rasterio.transform.from_bounds(142.0, 11.0, 143.0, 12.0, width, height)

    with rasterio.open(
        gebco_file,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs="EPSG:4326",
        nodata=-32768.0,
        transform=transform,
    ) as dst:
        dst.write(synth_gebco, 1)

    orig_open = rasterio.open

    def fake_rasterio_open(path, *args, **kwargs):
        if str(path) == fetcher.GEBCO_URL:
            return orig_open(gebco_file, *args, **kwargs)
        return orig_open(path, *args, **kwargs)

    monkeypatch.setattr(
        "core.services.bathymetry_service.rasterio.open", fake_rasterio_open
    )

    data = fetcher.fetch_elevation_for_bounds(bounds_global)
    assert data.size > 0
    assert np.nanmin(data) < -5000


def test_fetch_bathymetry(tmp_path, monkeypatch):
    """Backward compatibility wrapper testing both European and global pipelines."""
    (tmp_path / "eu").mkdir(parents=True, exist_ok=True)
    (tmp_path / "global").mkdir(parents=True, exist_ok=True)
    test_emodnet_fetch(tmp_path / "eu", monkeypatch)
    test_gebco_fetch_global(tmp_path / "global", monkeypatch)


def test_great_lakes_detection_and_surface_elevation():
    """Verify Great Lakes bounding box detection and IGLD85 surface datums."""
    fetcher = BathymetryFetcher()
    # Lake Superior, Lake Michigan, Lake Erie, Lake Ontario
    assert fetcher.is_in_great_lakes((-87.5, 47.0, -87.0, 47.5)) is True
    assert is_in_great_lakes((-87.7, 41.8, -87.5, 42.0)) is True
    assert is_in_great_lakes((5.32, 43.20, 5.34, 43.22)) is False

    assert get_great_lakes_surface_elevation((-87.5, 47.0, -87.0, 47.5)) == 183.2
    assert get_great_lakes_surface_elevation((-87.7, 41.8, -87.5, 42.0)) == 176.0
    assert get_great_lakes_surface_elevation((-81.7, 41.5, -81.5, 41.7)) == 173.5
    assert get_great_lakes_surface_elevation((-79.4, 43.5, -79.2, 43.7)) == 74.2


def test_great_lakes_merge_srtm_and_etopo():
    """Verify negative LWD depths in Great Lakes are converted to absolute elevation above sea level."""
    bounds = (
        -87.70,
        41.85,
        -87.55,
        41.95,
    )  # Chicago / Lake Michigan (surface = 176.0m)
    transform = rasterio.transform.from_bounds(*bounds, 4, 4)

    # Left two columns: Land at 185m; Right two columns: Lake Michigan surface at 176m in SRTM
    srtm = np.array(
        [
            [185.0, 185.0, 176.0, 176.0],
            [185.0, 185.0, 176.0, 176.0],
            [185.0, 185.0, 176.0, 176.0],
            [185.0, 185.0, 176.0, 176.0],
        ],
        dtype=np.float32,
    )
    # NOAA Great Lakes bathymetry: positive on land (+185m), negative depth below LWD (-20m) in water
    bathy = np.array(
        [
            [185.0, 185.0, -10.0, -20.0],
            [185.0, 185.0, -10.0, -20.0],
            [185.0, 185.0, -10.0, -20.0],
            [185.0, 185.0, -10.0, -20.0],
        ],
        dtype=np.float32,
    )
    land_mask = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )

    merged = merge_srtm_and_etopo(srtm, bathy, land_mask=land_mask, transform=transform)
    # Land remains 185.0m; water depth -10m -> 166.0m, -20m -> 156.0m (no 176m artificial cliff!)
    np.testing.assert_allclose(merged[:, 0], 185.0, atol=1e-3)
    np.testing.assert_allclose(merged[:, 2], 166.0, atol=1e-3)
    np.testing.assert_allclose(merged[:, 3], 156.0, atol=1e-3)
