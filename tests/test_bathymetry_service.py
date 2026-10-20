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


def test_swissbathy3d_lake_detection_and_surface_elevation():
    """Verify Swiss lake detection and surface datums for all 22 lakes."""
    from core.utils.swiss_topo_api import (
        SWISS_BATHY3D_LAKES,
        get_swiss_bathy3d_lakes_for_bounds,
        get_swiss_lake_surface_elevation,
    )

    assert len(SWISS_BATHY3D_LAKES) == 22

    # Rotsee (nested within Lucerne region)
    rotsee_bounds = (8.305, 47.068, 8.320, 47.074)
    lakes = get_swiss_bathy3d_lakes_for_bounds(rotsee_bounds)
    assert any(l["name"] == "Rotsee" for l in lakes)
    assert get_swiss_lake_surface_elevation(rotsee_bounds) == 419.0

    # Lake Geneva
    geneva_bounds = (6.50, 46.45, 6.55, 46.50)
    assert get_swiss_lake_surface_elevation(geneva_bounds) == 372.0

    # Lake Zurich
    zurich_bounds = (8.55, 47.30, 8.60, 47.35)
    assert get_swiss_lake_surface_elevation(zurich_bounds) == 406.0

    # Non-Swiss area (Paris)
    paris_bounds = (2.30, 48.85, 2.35, 48.90)
    assert get_swiss_bathy3d_lakes_for_bounds(paris_bounds) == []
    assert get_swiss_lake_surface_elevation(paris_bounds) is None


def test_download_swissbathy3d_tiles_offline(tmp_path, monkeypatch):
    """Test SwissBATHY3D tile extraction and reprojection using a synthetic local archive."""
    import zipfile

    from core.utils import swiss_topo_api

    # 1. Create a synthetic ESRI ASCII grid in EPSG:2056 covering Rotsee tile (2666, 1213)
    # xllcorner=2666000, yllcorner=1213000, cellsize=1, 20x20 pixels
    asc_text = (
        "ncols 20\n"
        "nrows 20\n"
        "xllcorner 2666000\n"
        "yllcorner 1213000\n"
        "cellsize 1\n"
        "nodata_value -3.4028234663852886e+38\n"
    )
    # Fill lake bed at 410m depth (surface is 419m)
    rows = []
    for _ in range(20):
        rows.append(" ".join(["410.5"] * 20))
    asc_content = (asc_text + "\n".join(rows)).encode("ascii")

    # 2. Package into a zip archive
    zip_path = tmp_path / "mock_rotsee.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("swissBATHY3D_CHLV95_LN02_2666_1213.asc", asc_content)

    # 3. Patch cache dir and lake entry
    cache_dir = tmp_path / "alti3d_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(swiss_topo_api, "ALTI3D_CACHE_DIR", cache_dir)

    # Point HttpRangeFile to read the local mock zip
    class LocalRangeFile(io.RawIOBase):
        def __init__(self, path):
            self._f = open(path, "rb")

        def readable(self):
            return True

        def seekable(self):
            return True

        def tell(self):
            return self._f.tell()

        def seek(self, offset, whence=0):
            return self._f.seek(offset, whence)

        def readinto(self, b):
            return self._f.readinto(b)

        def close(self):
            self._f.close()

    monkeypatch.setattr(
        swiss_topo_api,
        "HttpRangeFile",
        lambda url, timeout=20: LocalRangeFile(zip_path),
    )

    rotsee_bounds = (8.311, 47.069, 8.315, 47.072)
    tiles = swiss_topo_api.download_swissbathy3d_tiles(rotsee_bounds)
    assert len(tiles) >= 1
    assert all(p.endswith("_wgs84.tif") for p in tiles)

    # Verify GeoTIFF content
    with rasterio.open(tiles[0]) as src:
        assert src.crs.to_epsg() == 4326
        data = src.read(1)
        valid = data[data > -1000]
        assert len(valid) > 0
        np.testing.assert_allclose(valid.mean(), 410.5, atol=0.5)


def test_swiss_lake_slicing_water_level(monkeypatch):
    """Test ContourSlicingJob sets water_level for Swiss lakes when include_bathymetry=True."""
    from core.services.contour_generator import ContourSlicingJob

    rotsee_bounds = (8.305, 47.068, 8.320, 47.074)

    job = ContourSlicingJob(
        bounds=rotsee_bounds,
        height_per_layer=5.0,
        num_layers=3,
        simplify=0.0,
        substrate_size_mm=100.0,
        layer_thickness_mm=3.0,
        center=(8.3125, 47.071),
        smoothing=0,
        min_area=0.0,
        min_feature_width_mm=0.0,
        include_bathymetry=True,
    )

    # Mock DEM and base contours
    synth_dem = np.full((10, 10), 415.0, dtype=np.float32)
    transform = rasterio.transform.from_bounds(*rotsee_bounds, 10, 10)
    monkeypatch.setattr(
        job,
        "_fetch_and_prep_dem",
        lambda cb=None: (synth_dem, np.ma.masked_invalid(synth_dem), transform),
    )
    monkeypatch.setattr(
        job,
        "_generate_base_contours",
        lambda dem, masked, trans, cb=None: [
            {
                "elevation": 405.0,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "elevation": 415.0,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "elevation": 425.0,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
        ],
    )
    monkeypatch.setattr(
        job,
        "_run_post_processing",
        lambda contours, proj, utm, cx, cy, cb=None: contours,
    )

    contours = job.run()
    assert len(contours) == 3
    # Swiss lake surface elevation for Rotsee is 419.0m
    assert all(c.get("water_level") == 419.0 for c in contours)


def test_swiss_lake_water_mask_and_clamping(monkeypatch):
    """Test that Swiss lake bathymetry is preserved and lidar lake reflection is clamped."""
    from core.services.contour_generator import ContourSlicingJob

    rotsee_bounds = (8.305, 47.068, 8.320, 47.074)
    job = ContourSlicingJob(
        bounds=rotsee_bounds,
        height_per_layer=5.0,
        num_layers=3,
        simplify=0.0,
        substrate_size_mm=100.0,
        layer_thickness_mm=3.0,
        center=(8.3125, 47.071),
        smoothing=0,
        min_area=0.0,
        min_feature_width_mm=0.0,
        include_bathymetry=True,
        use_osm_water_mask=True,
    )

    # 10x10 synthetic DEM:
    # Top half (land): 430.0m
    # Bottom half (lake): 410.0m (bathymetry), plus one pixel with lidar reflection at 419.13m (datum is 419.0m)
    synth_dem = np.full((10, 10), 430.0, dtype=np.float32)
    synth_dem[5:, :] = 410.0
    synth_dem[5, 5] = 419.13  # Lidar lake surface reflection

    transform = rasterio.transform.from_bounds(*rotsee_bounds, 10, 10)

    # Mock DEM download and mosaic
    monkeypatch.setattr(
        "core.utils.dem.download_elevation_tiles_for_bounds",
        lambda b, include_bathymetry=False: ["mock_tile.tif"],
    )
    monkeypatch.setattr(
        "core.utils.dem.mosaic_and_crop",
        lambda paths, b: (synth_dem.copy(), transform),
    )

    # Mock OSM water mask: bottom half is water
    mock_mask = np.zeros((10, 10), dtype=bool)
    mock_mask[5:, :] = True
    monkeypatch.setattr(
        "core.utils.osm_tiles.fetch_osm_static_image",
        lambda b, zoom=15, provider="osm": (
            np.zeros((10, 10, 3), dtype=np.uint8),
            transform,
            "EPSG:4326",
        ),
    )
    monkeypatch.setattr(
        "core.utils.osm_tiles.generate_water_mask_from_tiles",
        lambda img, water_color: mock_mask,
    )

    elev, masked, tf = job._fetch_and_prep_dem()

    # Verify bathymetry was NOT wiped to NaN
    assert not np.isnan(elev[5:, :]).any()
    # Verify deep lake bed at 410.0m was preserved
    assert elev[9, 9] == 410.0
    # Verify lidar reflection at 419.13m was clamped below water surface (419.0m)
    assert elev[5, 5] == 418.9


def test_elevation_range_swiss_lake_no_bathy(monkeypatch):
    """Test ElevationRangeJob includes the lake surface elevation when bathymetry is off."""
    from core.services.elevation_service import ElevationRangeJob

    rotsee_bounds = (8.305, 47.068, 8.320, 47.074)
    job = ElevationRangeJob(rotsee_bounds, include_bathymetry=False)

    # Synthetic DEM where lowest pixel is 419.13m (lidar reflection off lake surface, datum is 419.0m)
    synth_dem = np.full((10, 10), 450.0, dtype=np.float32)
    synth_dem[5:, :] = 419.13
    transform = rasterio.transform.from_bounds(*rotsee_bounds, 10, 10)

    monkeypatch.setattr(
        "core.services.elevation_service.download_elevation_tiles_for_bounds",
        lambda b, include_bathymetry=False: ["mock_tile.tif"],
    )
    monkeypatch.setattr(
        "core.services.elevation_service.mosaic_and_crop",
        lambda paths, b, downsample_factor=10: (synth_dem, transform),
    )

    stats = job.run()
    # Min elevation should include 419.0m
    assert stats["min"] == 419.0
    assert stats["max"] == 450.0
