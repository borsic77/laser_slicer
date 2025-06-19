import logging
import os
import tempfile
from typing import Optional

import matplotlib.pyplot as plt
import osmnx as ox
import requests
from django.conf import settings
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _is_relation_bbox_too_large(rel_id, max_deg=2.0):
    """Check if the relation bounding box is larger than allowed.
    Args:
        rel_id (int): The OSM relation ID to check.
        max_deg (float): Maximum allowed bounding box size in degrees.
    Returns:
        bool: True if the bounding box is too large, False otherwise."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    relation({rel_id});
    out bb;
    """
    try:
        resp = requests.post(overpass_url, data={"data": query}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data["elements"]:
            bounds = data["elements"][0].get("bounds")
            if bounds:
                dlat = bounds["maxlat"] - bounds["minlat"]
                dlon = bounds["maxlon"] - bounds["minlon"]
                if dlat > max_deg or dlon > max_deg:
                    logger.warning(
                        f"Relation {rel_id} bounding box too large: "
                        f"{dlat:.2f}° lat, {dlon:.2f}° lon. Skipping detailed geometry."
                    )
                    return True
    except Exception as exc:
        logger.warning("Failed to get water relation bounding box: %s", exc)
    return False


def plot_fetched_water_polygon(
    water_polygon, debug_image_path, filename="fetched_water_polygon.png"
):
    """
    Plots the fetched (raw) water polygon in lon/lat coordinates.
    """
    if water_polygon is None:
        print("No water polygon to plot.")
        return

    fig, ax = plt.subplots()
    # Plot water polygon, handling both Polygon and MultiPolygon
    if water_polygon.geom_type == "Polygon":
        x, y = water_polygon.exterior.xy
        ax.plot(x, y, color="blue")
        for interior in water_polygon.interiors:
            xi, yi = interior.xy
            ax.plot(xi, yi, color="blue", linestyle="--")
    elif water_polygon.geom_type == "MultiPolygon":
        for poly in water_polygon.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="blue")
            for interior in poly.interiors:
                xi, yi = interior.xy
                ax.plot(xi, yi, color="blue", linestyle="--")
    ax.set_title("Fetched Water Polygon (raw, lon/lat)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.axis("equal")
    os.makedirs(debug_image_path, exist_ok=True)
    outpath = os.path.join(debug_image_path, filename)
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved water polygon plot to {outpath}")


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


def fetch_waterbody_polygon_osmnx(rel_id):
    """Fetch and return the full water multipolygon as GeoDataFrame using osmnx.
    Args:
        rel_id (int): The OSM relation ID for the water body.
    Returns:
        Optional[Polygon]: The largest water body polygon if found, otherwise None.
    """
    try:
        # Download the relation as OSM XML
        url = f"https://www.openstreetmap.org/api/0.6/relation/{rel_id}/full"
        response = requests.get(url)
        response.raise_for_status()
        osm_xml = response.content
        # Write the XML to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".osm", delete=False) as tmpfile:
            tmpfile.write(osm_xml)
            tmpfile_path = tmpfile.name
        # Parse the XML into a GeoDataFrame
        gdf = ox.features_from_xml(tmpfile_path)
        polys = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].geometry
        if polys.empty:
            return None
        largest = polys.iloc[polys.area.argmax()]
        return largest
    except Exception as exc:
        logger.warning("osmnx fetch failed: %s", exc)
        return None


def fetch_waterbody_polygon(
    lat: float, lon: float, radius_km: float = 5
) -> Optional[Polygon]:
    """Return the water polygon containing the point if any.
    Tries to use osmnx (which reconstructs complex multipolygons properly).
    Falls back to manual Overpass assembly if osmnx fails.
    """
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return None

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    # Step 1: Find the largest water relation containing this point
    query_ids = f"""
    [out:json][timeout:25];
    is_in({lat},{lon})->.a;
    relation(pivot.a)["natural"="water"];
    out ids;
    """
    try:
        resp_ids = requests.post(OVERPASS_URL, data={"data": query_ids}, timeout=30)
        resp_ids.raise_for_status()
        data_ids = resp_ids.json()
    except Exception as exc:
        logger.warning("Overpass request for water relation IDs failed: %s", exc)
        return None

    water_rel_ids = [
        el["id"] for el in data_ids.get("elements", []) if el["type"] == "relation"
    ]
    logger.debug(
        "Found %d water relations near point (%.6f, %.6f)", len(water_rel_ids), lon, lat
    )
    if not water_rel_ids:
        return None

    # Use the first relation (can be improved to choose largest later)
    rel_id = water_rel_ids[0]
    logger.debug("Using water relation ID: %d", rel_id)

    # Step 2: Only fetch osmnx geometry if bbox is not too large
    if not _is_relation_bbox_too_large(rel_id, max_deg=2.0):
        poly = fetch_waterbody_polygon_osmnx(rel_id)
        if poly is not None:
            plot_fetched_water_polygon(poly, settings.DEBUG_IMAGE_PATH)
            logger.debug("Fetched water polygon using osmnx")
            return poly
    else:
        logger.warning(
            f"Skipping osmnx fetch for relation {rel_id} due to large bbox; falling back."
        )

    # --- fallback: manual Overpass element parsing (legacy) ---
    query_geom = f"""
    [out:json][timeout:60];
    relation({rel_id});
    (._;>;);
    out geom;
    """
    try:
        resp_geom = requests.post(OVERPASS_URL, data={"data": query_geom}, timeout=60)
        resp_geom.raise_for_status()
        data_geom = resp_geom.json()
    except Exception as exc:
        logger.warning("Overpass request for water geometry failed: %s", exc)
        return None

    pt = Point(lon, lat)
    polys = []
    for element in data_geom.get("elements", []):
        poly = _element_to_polygon(element)
        if poly and poly.contains(pt):
            polys.append(poly)
    if polys:
        merged = unary_union(polys)
        plot_fetched_water_polygon(merged, settings.DEBUG_IMAGE_PATH)
        if merged.geom_type == "Polygon":
            logger.debug("Fetched water polygon (manual Overpass fallback)")
            return merged
        if merged.geom_type == "MultiPolygon":
            logger.debug(
                "Fetched MultiPolygon water body with %d parts (manual Overpass fallback)",
                len(merged.geoms),
            )
            return max(merged.geoms, key=lambda p: p.area)
    return None
