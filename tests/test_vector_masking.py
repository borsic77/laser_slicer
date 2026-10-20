import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import Polygon

from core.services.osm_service import fetch_coastline_mask

matplotlib.use("Agg")


def test_la_gomera_mask(tmp_path, monkeypatch):
    bounds = (-17.35, 28.08, -17.31, 28.11)
    res = 0.001
    width = int((bounds[2] - bounds[0]) / res)
    height = int((bounds[3] - bounds[1]) / res)
    shape = (height, width)
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    gomera_land = Polygon(
        [
            (-17.35, 28.08),
            (-17.33, 28.08),
            (-17.33, 28.11),
            (-17.35, 28.11),
            (-17.35, 28.08),
        ]
    )
    gdf = gpd.GeoDataFrame({"geometry": [gomera_land]}, crs="EPSG:4326")

    monkeypatch.setattr(
        "core.services.osm_service.ox.features_from_polygon", lambda *a, **k: gdf
    )
    monkeypatch.setattr(
        "core.services.osm_service.is_land_central_point", lambda *a, **k: True
    )

    mask = fetch_coastline_mask(bounds, shape, transform)
    output_dir = tmp_path / "data" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_dir / "test_mask_gomera.png", mask, cmap="Greys")

    assert mask.shape == shape
    assert np.sum(mask) > 0
    assert np.sum(mask) < mask.size


def test_deep_ocean_mask(monkeypatch):
    bounds = (-30.0, 30.0, -29.99, 30.01)
    res = 0.001
    width, height = 10, 10
    shape = (height, width)
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    def fake_features(*a, **k):
        raise Exception("No matching features")

    monkeypatch.setattr(
        "core.services.osm_service.ox.features_from_polygon", fake_features
    )
    monkeypatch.setattr(
        "core.services.osm_service.is_land_central_point", lambda *a, **k: False
    )

    mask = fetch_coastline_mask(bounds, shape, transform)
    assert mask.shape == shape
    assert np.sum(mask) == 0


def test_land_only_mask(monkeypatch):
    bounds = (8.54, 47.37, 8.55, 47.38)
    res = 0.0001
    width, height = 10, 10
    shape = (height, width)
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    def fake_features(*a, **k):
        raise Exception("No matching features")

    monkeypatch.setattr(
        "core.services.osm_service.ox.features_from_polygon", fake_features
    )
    monkeypatch.setattr(
        "core.services.osm_service.is_land_central_point", lambda *a, **k: True
    )

    mask = fetch_coastline_mask(bounds, shape, transform)
    assert mask.shape == shape
    assert np.sum(mask) == mask.size


def test_vector_masking_san_francisco(monkeypatch):
    """Test vector coastline masking for San Francisco Peninsula with mocked Overpass data."""
    bounds = (-122.52, 37.70, -122.35, 37.83)
    res = 0.001
    width = int((bounds[2] - bounds[0]) / res)
    height = int((bounds[3] - bounds[1]) / res)
    shape = (height, width)
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    sf_peninsula = Polygon(
        [
            (-122.51, 37.71),
            (-122.38, 37.71),
            (-122.38, 37.81),
            (-122.51, 37.81),
            (-122.51, 37.71),
        ]
    )
    gdf = gpd.GeoDataFrame({"geometry": [sf_peninsula]}, crs="EPSG:4326")

    monkeypatch.setattr(
        "core.services.osm_service.ox.features_from_polygon", lambda *a, **k: gdf
    )
    monkeypatch.setattr(
        "core.services.osm_service.is_land_central_point", lambda *a, **k: True
    )

    mask = fetch_coastline_mask(bounds, shape, transform)
    assert mask.shape == shape
    assert np.sum(mask) > 0
    assert np.sum(mask) < mask.size
