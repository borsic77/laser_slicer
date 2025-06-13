import logging
import math
from typing import Optional

import requests
from shapely.geometry import MultiPolygon, Polygon, Point
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _element_to_polygon(element) -> Optional[Polygon]:
    """Convert an Overpass element to a Shapely polygon."""
    try:
        if element.get("type") == "way" and element.get("geometry"):
            coords = [(n["lon"], n["lat"]) for n in element["geometry"]]
            if len(coords) < 3:
                return None
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            poly = Polygon(coords)
            return poly if poly.is_valid else None
        if element.get("type") == "relation" and element.get("members"):
            polys = []
            for mem in element["members"]:
                if mem.get("role") == "outer" and mem.get("geometry"):
                    coords = [(n["lon"], n["lat"]) for n in mem["geometry"]]
                    if len(coords) < 3:
                        continue
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    poly = Polygon(coords)
                    if poly.is_valid:
                        polys.append(poly)
            if polys:
                merged = unary_union(polys)
                if merged.geom_type == "Polygon":
                    return merged
                if merged.geom_type == "MultiPolygon":
                    return max(merged.geoms, key=lambda p: p.area)
    except Exception as exc:
        logger.warning("Failed to parse overpass element: %s", exc)
    return None


def fetch_waterbody_polygon(lat: float, lon: float, radius_km: float = 5) -> Optional[Polygon]:
    """Return the water polygon containing the point if any."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None

    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.32 * math.cos(math.radians(lat)) or 1)
    south = lat - lat_radius
    north = lat + lat_radius
    west = lon - lon_radius
    east = lon + lon_radius

    query = f"""
    [out:json][timeout:25];
    (
      way["natural"="water"]({south},{west},{north},{east});
      relation["natural"="water"]({south},{west},{north},{east});
      way["waterway"="riverbank"]({south},{west},{north},{east});
      relation["waterway"="riverbank"]({south},{west},{north},{east});
    );
    out geom;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Overpass request failed: %s", exc)
        return None

    pt = Point(lon, lat)
    for element in data.get("elements", []):
        poly = _element_to_polygon(element)
        if poly and poly.contains(pt):
            return poly
    return None

