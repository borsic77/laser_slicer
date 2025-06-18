import geopandas as gpd
from shapely.geometry import LineString, Polygon
import types
import sys

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

from core.utils.osm_features import fetch_roads, fetch_buildings


def test_fetch_roads(monkeypatch):
    gdf = gpd.GeoDataFrame({"geometry": [LineString([(0,0),(1,0)])]})
    monkeypatch.setattr("osmnx.geometries_from_polygon", lambda p, tags=None: gdf)
    result = fetch_roads((0,0,1,1))
    assert result.geom_type == "MultiLineString"
    assert len(result.geoms) == 1


def test_fetch_buildings(monkeypatch):
    gdf = gpd.GeoDataFrame({"geometry": [Polygon([(0,0),(1,0),(1,1),(0,0)])]})
    monkeypatch.setattr("osmnx.geometries_from_polygon", lambda p, tags=None: gdf)
    result = fetch_buildings((0,0,1,1))
    assert result.geom_type == "MultiPolygon"
    assert len(result.geoms) == 1
