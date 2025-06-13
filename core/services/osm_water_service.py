import logging
from typing import Tuple, Optional

import requests
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
import numpy as np
from rasterio.features import geometry_mask

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.slicer import mosaic_and_crop

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


class OSMWaterService:
    """Fetch OSM water polygons around a point and compute average elevation."""

    def __init__(self, lat: float, lon: float, radius: int = 300) -> None:
        self.lat = lat
        self.lon = lon
        self.radius = radius

    # --- Public API -----------------------------------------------------
    def fetch_water_polygon(self) -> Optional[Polygon | MultiPolygon]:
        query = f"""
[out:json];
(
  way(around:{self.radius},{self.lat},{self.lon})["natural"="water"];
  relation(around:{self.radius},{self.lat},{self.lon})["natural"="water"];
  way(around:{self.radius},{self.lat},{self.lon})["waterway"="riverbank"];
  relation(around:{self.radius},{self.lat},{self.lon})["waterway"="riverbank"];
);
out geom;
"""
        try:
            resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Overpass request failed: %s", exc)
            return None
        data = resp.json()
        elems = data.get("elements", [])
        point = Point(self.lon, self.lat)
        geoms = []
        for el in elems:
            geom = self._element_to_geometry(el)
            if geom is not None and geom.contains(point):
                geoms.append(geom)
        if not geoms:
            return None
        return unary_union(geoms)

    def average_elevation(self, polygon: Polygon | MultiPolygon) -> Optional[float]:
        if polygon.is_empty:
            return None
        lon_min, lat_min, lon_max, lat_max = polygon.bounds
        paths = download_srtm_tiles_for_bounds((lon_min, lat_min, lon_max, lat_max))
        elev, transform = mosaic_and_crop(paths, (lon_min, lat_min, lon_max, lat_max))
        mask = geometry_mask([mapping(polygon)], out_shape=elev.shape, transform=transform, invert=True)
        data = np.ma.array(elev, mask=~mask)
        valid = np.ma.masked_where(~np.isfinite(data) | (data <= -32768), data)
        if valid.count() == 0:
            return None
        return float(valid.mean())

    # ------------------------------------------------------------------
    def _element_to_geometry(self, elem) -> Optional[Polygon | MultiPolygon]:
        try:
            if elem["type"] == "way" and "geometry" in elem:
                coords = [(n["lon"], n["lat"]) for n in elem["geometry"]]
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                return Polygon(coords)
            if elem["type"] == "relation" and elem.get("members"):
                polys = []
                for mem in elem["members"]:
                    if mem.get("role") in ("outer", "") and "geometry" in mem:
                        c = [(n["lon"], n["lat"]) for n in mem["geometry"]]
                        if c and c[0] != c[-1]:
                            c.append(c[0])
                        polys.append(Polygon(c))
                if polys:
                    return MultiPolygon(polys)
        except Exception as exc:
            logger.debug("Error parsing element geometry: %s", exc)
        return None

    def run(self) -> Optional[Tuple[dict, float]]:
        poly = self.fetch_water_polygon()
        if poly is None:
            return None
        elevation = self.average_elevation(poly)
        if elevation is None:
            return None
        return mapping(poly), elevation
