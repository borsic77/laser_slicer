# New service to query OpenStreetMap for water polygons
import logging
from typing import Iterable, List, Optional

import requests
from shapely.geometry import MultiPolygon, Point, Polygon

logger = logging.getLogger(__name__)


class OSMWaterService:
    """Query Overpass API for water polygons covering a point."""

    def __init__(
        self,
        *,
        radii: Optional[Iterable[int]] = None,
        max_radius: Optional[int] = None,
        overpass_url: str = "https://overpass-api.de/api/interpreter",
    ) -> None:
        if radii is None:
            if max_radius is not None:
                default = [300, 1000, 5000]
                self.radii = [r for r in default if r <= max_radius] or [max_radius]
            else:
                self.radii = [300, 1000, 5000]
        else:
            self.radii = list(radii)
        self.overpass_url = overpass_url
        self.session = requests.Session()

    def _build_query(self, lat: float, lon: float, radius: int) -> str:
        return f"""
[out:json];
(
  way[\"natural\"=\"water\"](around:{radius},{lat},{lon});
  relation[\"natural\"=\"water\"](around:{radius},{lat},{lon});
  way[\"waterway\"=\"riverbank\"](around:{radius},{lat},{lon});
  relation[\"waterway\"=\"riverbank\"](around:{radius},{lat},{lon});
);
out geom;
"""

    def _query_overpass(self, query: str) -> dict:
        resp = self.session.post(self.overpass_url, data=query, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _element_to_polygons(element: dict) -> List[Polygon]:
        polys: List[Polygon] = []
        if element.get("geometry"):
            coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
            if coords and coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                polys.append(Polygon(coords))
            except Exception as exc:  # pragma: no cover - geom errors
                logger.debug("Failed to build polygon: %s", exc)
        elif element.get("members"):
            outers: List[Polygon] = []
            inners: List[Polygon] = []
            for m in element["members"]:
                if m.get("geometry"):
                    coords = [(p["lon"], p["lat"]) for p in m["geometry"]]
                    if coords and coords[0] != coords[-1]:
                        coords.append(coords[0])
                    poly = Polygon(coords)
                    if m.get("role") == "outer":
                        outers.append(poly)
                    elif m.get("role") == "inner":
                        inners.append(poly)
            for poly in outers:
                if inners:
                    poly = poly.difference(MultiPolygon(inners))
                polys.append(poly)
        return polys

    def fetch_water_polygon(self, lat: float, lon: float) -> Optional[Polygon | MultiPolygon]:
        """Return a water polygon covering the given point or ``None``."""
        point = Point(lon, lat)
        for radius in self.radii:
            try:
                data = self._query_overpass(self._build_query(lat, lon, radius))
            except requests.RequestException as exc:  # pragma: no cover - network
                logger.warning("Overpass request failed: %s", exc)
                continue
            elements = data.get("elements", [])
            polygons: List[Polygon] = []
            for el in elements:
                polygons.extend(self._element_to_polygons(el))
            for poly in polygons:
                if poly.is_valid and poly.contains(point):
                    return poly
        return None
