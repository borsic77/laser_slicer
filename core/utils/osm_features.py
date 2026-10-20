import threading
from collections import OrderedDict
from pathlib import Path

import osmnx as ox
from django.conf import settings
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.ops import unary_union

from core.utils.cache_ops import enforce_cache_limit, touch_cache_file

__all__ = ["fetch_buildings", "fetch_roads", "fetch_waterways"]

_OVERPASS_LOCK = threading.Lock()
_DEFAULT_OVERPASS_URL = "https://overpass-api.de/api"


def _bounds_polygon(bounds: tuple[float, float, float, float]) -> Polygon:
    lon_min, lat_min, lon_max, lat_max = bounds
    return Polygon(
        [
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_max),
            (lon_max, lat_min),
        ]
    )


def _find_cached_timeout(poly: Polygon, tags: dict) -> int | None:
    """Return the timeout setting (180, 20, or 15) that already has a local OSMnx cache hit."""
    try:
        import osmnx._http as ox_http
        import osmnx._overpass as ox_ov
        import requests

        coord_strs = ox_ov._make_overpass_polygon_coord_strs(poly)
        if not coord_strs:
            return None
        url = _DEFAULT_OVERPASS_URL.rstrip("/") + "/interpreter"
        for timeout_val in (180, 20, 15):
            ox.settings.requests_timeout = timeout_val
            q = ox_ov._create_overpass_features_query(coord_strs[0], tags)
            prep = str(
                requests.Request("GET", url, params=OrderedDict(data=q)).prepare().url
            )
            cache_path = ox_http._check_cache(prep)
            if cache_path is not None:
                touch_cache_file(cache_path)
                return timeout_val
    except Exception:
        pass
    return None


def _features_from_polygon(poly, tags):
    """
    Fetch features from OSM within a polygon using OSMnx.
    This function handles both OSMnx < 2.0 and OSMnx ≥ 2.0 versions.
    Args:
        poly (Polygon): The polygon to query.
        tags (dict): Tags to filter the features.
    Returns:
        GeoDataFrame: A GeoDataFrame containing the features.
    """
    cache_dir = Path(getattr(settings, "OSM_CACHE_DIR", Path("data/osm_cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _OVERPASS_LOCK:
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(cache_dir)
        ox.settings.overpass_rate_limit = False
        ox.settings.overpass_url = _DEFAULT_OVERPASS_URL

        cached_timeout = _find_cached_timeout(poly, tags)
        ox.settings.requests_timeout = (
            cached_timeout if cached_timeout is not None else 15
        )

        try:
            if hasattr(ox, "geometries_from_polygon"):
                gdf = ox.geometries_from_polygon(poly, tags=tags)
            else:  # OSMnx ≥ 2.0
                gdf = ox.features_from_polygon(poly, tags=tags)
        except Exception as exc:
            if "No matching features" in str(exc):
                import geopandas as gpd

                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            raise
        finally:
            enforce_cache_limit(
                cache_dir,
                int(getattr(settings, "OSM_CACHE_MAX_MB", 512)) * 1024 * 1024,
            )

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def fetch_roads(
    bounds: tuple[float, float, float, float],
) -> dict[str, MultiLineString]:
    """Fetch road geometries grouped by ``highway`` type within bounds."""

    poly = _bounds_polygon(bounds)
    gdf = _features_from_polygon(poly, {"highway": True})

    features: dict[str, list] = {}
    for _, row in gdf.iterrows():
        geom = row.geometry
        road_types = row.get("highway")
        if geom is None or geom.is_empty or not road_types:
            continue
        if not isinstance(road_types, list):
            road_types = [road_types]

        def _add_geom(g):
            gtype = g.geom_type
            if gtype in ("LineString", "MultiLineString"):
                for rt in road_types:
                    features.setdefault(rt, []).append(g)
            elif gtype in ("Polygon", "MultiPolygon"):
                for rt in road_types:
                    features.setdefault(rt, []).append(g.boundary)
            elif hasattr(g, "geoms"):
                for part in g.geoms:
                    _add_geom(part)

        _add_geom(geom)

    result: dict[str, MultiLineString] = {}
    for rtype, geoms in features.items():
        if not geoms:
            continue
        merged = unary_union(geoms)
        if isinstance(merged, LineString):
            result[rtype] = MultiLineString([merged])
        elif isinstance(merged, MultiLineString):
            result[rtype] = merged
        elif hasattr(merged, "geoms"):
            segments = [
                g for g in merged.geoms if isinstance(g, (LineString, MultiLineString))
            ]
            if segments:
                merge2 = unary_union(segments)
                if isinstance(merge2, LineString):
                    result[rtype] = MultiLineString([merge2])
                elif isinstance(merge2, MultiLineString):
                    result[rtype] = merge2
    return result


def fetch_waterways(bounds: tuple[float, float, float, float]) -> MultiLineString:
    """Fetch waterway geometries within bounds from OSM.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    MultiLineString
        Collection of river/stream/canal centerlines.
    """

    poly = _bounds_polygon(bounds)
    gdf = _features_from_polygon(poly, {"waterway": ["river", "stream", "canal"]})
    lines = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        gtype = geom.geom_type
        if gtype in ("LineString", "MultiLineString"):
            lines.append(geom)
        elif gtype in ("Polygon", "MultiPolygon"):
            lines.append(geom.boundary)
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                if part.geom_type in ("LineString", "MultiLineString"):
                    lines.append(part)
                elif part.geom_type in ("Polygon", "MultiPolygon"):
                    lines.append(part.boundary)
    if not lines:
        return MultiLineString([])
    merged = unary_union(lines)
    if isinstance(merged, LineString):
        return MultiLineString([merged])
    if isinstance(merged, MultiLineString):
        return merged
    if hasattr(merged, "geoms"):
        segments = [
            g for g in merged.geoms if isinstance(g, (LineString, MultiLineString))
        ]
        if segments:
            merge2 = unary_union(segments)
            if isinstance(merge2, LineString):
                return MultiLineString([merge2])
            if isinstance(merge2, MultiLineString):
                return merge2
    return MultiLineString([])


def fetch_buildings(bounds: tuple[float, float, float, float]) -> MultiPolygon:
    """Fetch building footprints within bounds from OSM."""
    poly = _bounds_polygon(bounds)
    gdf = _features_from_polygon(poly, tags={"building": True})
    polys = []
    for geom in gdf.geometry:
        if geom.is_empty:
            continue
        gtype = geom.geom_type
        if gtype == "Polygon":
            polys.append(geom)
        elif gtype == "MultiPolygon":
            polys.extend(list(geom.geoms))
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                if part.geom_type == "Polygon":
                    polys.append(part)
                elif part.geom_type == "MultiPolygon":
                    polys.extend(list(part.geoms))
    if not polys:
        return MultiPolygon([])
    merged = unary_union(polys)
    if isinstance(merged, MultiPolygon):
        return merged
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    if hasattr(merged, "geoms"):
        polygons = [
            p for p in merged.geoms if p.geom_type in ("Polygon", "MultiPolygon")
        ]
        if polygons:
            unioned = unary_union(polygons)
            if isinstance(unioned, Polygon):
                return MultiPolygon([unioned])
            if isinstance(unioned, MultiPolygon):
                return unioned
    return MultiPolygon([])


def normalize_road_geometry(
    clipped: GeometryCollection | LineString | MultiLineString,
) -> MultiLineString | None:
    """
    Normalize road geometry: always return a MultiLineString or None.
    Args:
        clipped (GeometryCollection | LineString | MultiLineString): The clipped road geometry.
    Returns:
        MultiLineString | None: Normalized road geometry or None if empty.
    """

    if clipped.is_empty:
        return None
    if clipped.geom_type == "LineString":
        return MultiLineString([clipped])
    elif clipped.geom_type == "MultiLineString":
        return clipped
    elif clipped.geom_type == "GeometryCollection":
        lines = [
            g for g in clipped.geoms if g.geom_type in ("LineString", "MultiLineString")
        ]
        if not lines:
            return None
        merged = unary_union(lines)
        if merged.geom_type == "LineString":
            return MultiLineString([merged])
        return merged
    return None


def normalize_waterway_geometry(
    clipped: GeometryCollection | LineString | MultiLineString,
) -> MultiLineString | None:
    """Normalize waterway geometry to a MultiLineString or ``None``."""

    return normalize_road_geometry(clipped)


def normalize_building_geometry(
    clipped: GeometryCollection | Polygon | MultiPolygon,
) -> MultiPolygon | None:
    """
    Normalize building geometry: always return a MultiPolygon or None.
    Args:
        clipped (GeometryCollection | Polygon | MultiPolygon): The clipped building geometry.
    Returns:
        MultiPolygon | None: Normalized building geometry or None if empty.
    """

    if clipped.is_empty:
        return None
    if clipped.geom_type == "Polygon":
        return MultiPolygon([clipped])
    elif clipped.geom_type == "MultiPolygon":
        return clipped
    elif clipped.geom_type == "GeometryCollection":
        polys = [g for g in clipped.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        merged = unary_union(polys)
        if merged.geom_type == "Polygon":
            return MultiPolygon([merged])
        return merged
    return None
