import logging

import numpy as np
import osmnx as ox
import rasterio
import rasterio.features
import requests
from django.conf import settings
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import polygonize, unary_union

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def is_land_central_point(lat, lon):
    """
    Check if a point is on land using Overpass is_in query.
    We look for specific land tags and avoid territorial water relations.
    """
    query = f"""
    [out:json][timeout:15];
    is_in({lat},{lon})->.a;
    (
      area.a["place"~"island|islet|continent|city|town|village"];
      area.a["natural"~"wood|land|grass|scrub|heath"];
      area.a["landuse"];
    );
    out count;
    """
    try:
        resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        count = int(data.get("elements", [{}])[0].get("tags", {}).get("total", 0))
        # If we found land-use or specific land places, it's land.
        return count > 0
    except Exception as e:
        logger.warning(f"Overpass is_in check failed: {e}")
        return None


def fetch_coastline_mask(bounds, shape, transform):
    """
    Fetch coastline vectors from OSM and rasterize into a binary mask.
    1 = Land, 0 = Sea.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    bbox_poly = box(lon_min, lat_min, lon_max, lat_max)

    try:
        # Step 1: Fetch coastline lines using the bbox polygon
        # This avoids the bbox tuple order issues
        gdf_lines = None
        try:
            gdf_lines = ox.features_from_polygon(
                bbox_poly, tags={"natural": "coastline"}
            )
        except Exception as e:
            if "No matching features" in str(e):
                logger.info("No coastline lines found in bbox.")
            else:
                logger.warning(f"OSMnx fetch coastline lines failed: {e}")

        # Step 2: Fetch land polygons
        gdf_polys = None
        try:
            tags_land = {"place": ["island", "islet", "continent"], "natural": ["land"]}
            gdf_polys = ox.features_from_polygon(bbox_poly, tags=tags_land)
        except Exception as e:
            if "No matching features" in str(e):
                logger.info("No land polygons found in bbox.")
            else:
                logger.warning(f"OSMnx land polygon fetch failed: {e}")

        polygons = []

        # Add polygons from gdf_polys
        if gdf_polys is not None and not gdf_polys.empty:
            for geom in gdf_polys.geometry:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    clipped = geom.intersection(bbox_poly)
                    if not clipped.is_empty:
                        polygons.append(clipped)

        # Reconstruct polygons from lines using polygonize
        if gdf_lines is not None and not gdf_lines.empty:
            lines = []
            for geom in gdf_lines.geometry:
                if isinstance(geom, LineString):
                    lines.append(geom)
                elif hasattr(geom, "geoms"):  # MultiLineString
                    lines.extend(list(geom.geoms))

            if lines:
                # We need the boundary of the bbox to close polygons that exit the box
                boundary = bbox_poly.boundary
                all_lines = unary_union(lines + [boundary])
                result_polys = list(polygonize(all_lines))

                # Filter result_polys: keep only those that are land
                for p in result_polys:
                    # Avoid duplications with Step 2 (though unary_union will handle it later)
                    # We check the representative point
                    rep_pt = p.representative_point()
                    if is_land_central_point(rep_pt.y, rep_pt.x):
                        polygons.append(p)

        if not polygons:
            # Fallback for deep sea or deep land
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            if is_land_central_point(center_lat, center_lon):
                logger.info("No local features. Point is on land. Assuming all land.")
                return np.ones(shape, dtype=np.uint8)
            else:
                logger.info("No local features. Point is sea. Assuming all sea.")
                return np.zeros(shape, dtype=np.uint8)

        # Merge all land polygons
        land_union = unary_union(polygons)
        land_union = land_union.intersection(bbox_poly)

        # Rasterize
        mask = rasterio.features.rasterize(
            [(land_union, 1)],
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8,
        )

        return mask

    except Exception as e:
        logger.error(f"Error fetching coastline mask: {e}", exc_info=True)
        return np.zeros(shape, dtype=np.uint8)
