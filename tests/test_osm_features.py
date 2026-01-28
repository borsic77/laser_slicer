import sys
import types

import geopandas as gpd
from shapely.geometry import LineString, Polygon

# Insert a dummy 'celery' module so importing core does not fail
celery_mod = types.ModuleType("celery")


class DummyCelery:
    def __init__(self, *args, **kwargs):
        pass

    def config_from_object(self, *args, **kwargs):
        pass

    def autodiscover_tasks(self, *args, **kwargs):
        pass

    def task(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


celery_mod.Celery = DummyCelery
sys.modules.setdefault("celery", celery_mod)

from core.utils.osm_features import (
    fetch_buildings,
    fetch_roads,
    fetch_waterways,
)


def test_fetch_roads(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 0)])],
            "highway": ["residential"],
        },
        crs="EPSG:4326",
    )
    monkeypatch.setattr("osmnx.features_from_polygon", lambda p, tags=None: gdf)
    result = fetch_roads((0, 0, 1, 1))
    assert isinstance(result, dict) and "residential" in result
    geom = result["residential"]
    assert geom.geom_type == "MultiLineString" and len(geom.geoms) == 1


def test_fetch_buildings(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])]}, crs="EPSG:4326"
    )
    monkeypatch.setattr("osmnx.features_from_polygon", lambda p, tags=None: gdf)
    result = fetch_buildings((0, 0, 1, 1))
    assert result.geom_type == "MultiPolygon"
    assert len(result.geoms) == 1


def test_fetch_waterways(monkeypatch):
    gdf = gpd.GeoDataFrame(
        {"geometry": [LineString([(0, 0), (1, 1)])]}, crs="EPSG:4326"
    )
    monkeypatch.setattr("osmnx.features_from_polygon", lambda p, tags=None: gdf)
    result = fetch_waterways((0, 0, 1, 1))
    assert result.geom_type == "MultiLineString"
    assert len(result.geoms) == 1
