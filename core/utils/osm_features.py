import osmnx as ox
from shapely.geometry import Polygon, MultiLineString, MultiPolygon, LineString
from shapely.ops import unary_union

__all__ = ["fetch_roads", "fetch_buildings"]


def _bounds_polygon(bounds: tuple[float, float, float, float]) -> Polygon:
    lon_min, lat_min, lon_max, lat_max = bounds
    return Polygon([
        (lon_min, lat_min),
        (lon_min, lat_max),
        (lon_max, lat_max),
        (lon_max, lat_min),
    ])


def fetch_roads(bounds: tuple[float, float, float, float]) -> MultiLineString:
    """Fetch road geometries within bounds from OSM."""
    poly = _bounds_polygon(bounds)
    gdf = ox.geometries_from_polygon(poly, tags={"highway": True})
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
        segments = [g for g in merged.geoms if isinstance(g, (LineString, MultiLineString))]
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
    gdf = ox.geometries_from_polygon(poly, tags={"building": True})
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
        polygons = [p for p in merged.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
        if polygons:
            unioned = unary_union(polygons)
            if isinstance(unioned, Polygon):
                return MultiPolygon([unioned])
            if isinstance(unioned, MultiPolygon):
                return unioned
    return MultiPolygon([])

