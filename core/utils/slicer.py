import logging
import math
import os
from collections import defaultdict
from tkinter import N
from typing import List, Tuple

import elevation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import shapely
from django.conf import settings
from filelock import FileLock
from rasterio.merge import merge
from rasterio.windows import from_bounds
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Polygon,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.validation import make_valid
from shapely.ops import transform, unary_union

# Use headless matplotlib
matplotlib.use("Agg")

TILE_CACHE_DIR = settings.TILE_CACHE_DIR
DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
DEBUG = settings.DEBUG
if DEBUG:
    os.makedirs(DEBUG_IMAGE_PATH, exist_ok=True)
logger = logging.getLogger(__name__)


def round_affine(
    transform: rasterio.Affine, precision: float = 1e-4
) -> rasterio.Affine:
    """Rounds the components of an affine transform to a given precision.

    Args:
        transform (rasterio.Affine): The affine transform to round.
        precision (float): The precision to round to.
    returns:
        rasterio.Affine: The rounded affine transform.
    """
    return rasterio.Affine(
        *(round(val, int(-math.log10(precision))) for val in transform)
    )


def clean_geometry(geom: BaseGeometry) -> BaseGeometry | None:
    """Cleans a geometry by attempting to fix invalid shapes.

    Applies make_valid to fix topology issues and
    removes empty or zero-area geometries.

    Args:
        geom (BaseGeometry): A Shapely geometry object.

    Returns:
        BaseGeometry | None: A cleaned geometry if valid, otherwise None.
    """
    geom = make_valid(geom)
    if geom.is_empty or geom.area == 0:
        return None
    return orient(geom)


def save_debug_contour_polygon(polygon, level: float, filename: str) -> None:
    """Saves a debug image of a single contour polygon at a given elevation.

    Args:
        polygon (Polygon or MultiPolygon): The geometry to plot.
        level (float): The elevation level of the contour.
        filename (str): The base filename to use for the saved image.
    """
    fig, ax = plt.subplots()
    if polygon.is_empty or not polygon.is_valid:
        return
    if polygon.geom_type == "Polygon":
        x, y = polygon.exterior.xy
        ax.plot(x, y)
    elif polygon.geom_type == "MultiPolygon":
        for p in polygon.geoms:
            x, y = p.exterior.xy
            ax.plot(x, y)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(DEBUG_IMAGE_PATH, f"{filename}_elev_{level}.png"))
    plt.close(fig)


def mosaic_and_crop(
    tif_paths: List[str], bounds: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, rasterio.Affine]:
    """Merges and crops SRTM tiles to the bounding box.

    Args:
        tif_paths (list): List of .hgt.gz or GeoTIFF paths.
        bounds (tuple): Bounding box (lon_min, lat_min, lon_max, lat_max).

    Returns:
        tuple: Cropped elevation array and its affine transform.
    """
    src_files = [
        rasterio.open(f"/vsigzip/{p}") if p.endswith(".gz") else rasterio.open(p)
        for p in tif_paths
    ]

    # Merge to a single raster
    mosaic, transform = merge(src_files)
    merged_bounds = rasterio.transform.array_bounds(
        mosaic.shape[1], mosaic.shape[2], transform
    )
    logger.debug(f"Merged raster bounds: {merged_bounds}")
    logger.debug(f"Requested crop bounds: {bounds}")
    logger.debug(f"Merged shape: {mosaic.shape}, transform: {transform}")
    # Unpack bounds and ensure lat_min < lat_max
    lon_min, lat_min, lon_max, lat_max = bounds
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    bounds = (lon_min, lat_min, lon_max, lat_max)

    # Check raster orientation
    if transform.e > 0:
        logger.warning(
            "Unexpected raster orientation (north-up). SRTM tiles usually have north-down orientation."
        )

    # Clip using bounds, with error handling
    try:
        window = from_bounds(*bounds, transform=transform)
    except ValueError as e:
        logger.error(
            f"Failed to compute raster window from bounds {bounds} with given transform: {transform}"
        )
        raise
    row_off = int(window.row_off)
    row_end = row_off + int(window.height)
    col_off = int(window.col_off)
    col_end = col_off + int(window.width)

    clipped = mosaic[:, row_off:row_end, col_off:col_end]

    cropped_transform = round_affine(rasterio.windows.transform(window, transform))

    for src in src_files:
        src.close()

    return clipped[0], cropped_transform


def walk_bbox_between(
    coords, start_idx: int, end_idx: int, direction: str = "cw"
) -> list:
    """Walks the bbox coordinates circularly from end to start.

    Args:
        coords (list): List of coordinate tuples.
        start_idx (int): Index to start from.
        end_idx (int): Index to end at.
        direction (str): 'cw' for clockwise, 'ccw' for counterclockwise.

    Returns:
        list: Subset of coords between indices.
    """
    n = len(coords)
    if direction == "cw":
        if start_idx >= end_idx:
            return coords[end_idx : start_idx + 1]
        else:
            return coords[end_idx:] + coords[: start_idx + 1]
    else:  # ccw
        if start_idx <= end_idx:
            return coords[start_idx : end_idx + 1][::-1]
        else:
            return (coords[start_idx:] + coords[: end_idx + 1])[::-1]


def is_almost_closed(line: LineString, tolerance: float = 1e-8) -> bool:
    """Checks if a LineString is nearly closed within a tolerance.

    Args:
        line (LineString): The line geometry to check.
        tolerance (float): Maximum allowable gap length.

    Returns:
        bool: True if nearly closed, else False.
    """
    return (
        line.coords[0] != line.coords[-1]
        and LineString([line.coords[0], line.coords[-1]]).length < tolerance
    )


def _prepare_meshgrid(
    elevation_data: np.ndarray, transform: rasterio.Affine
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate longitude and latitude meshgrids from raster shape and affine transform.

    Converts pixel-based indices to geospatial coordinates using Rasterio's transform,
    with correct handling of row-major indexing.

    Args:
        elevation_data (np.ndarray): 2D elevation raster array (shape: [rows, cols]).
        transform (rasterio.Affine): Affine transformation matrix for the raster.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 2D arrays (lon, lat), each matching
        the shape of `elevation_data`, containing geographic coordinates.
    """
    ny, nx = elevation_data.shape
    y = np.arange(ny)
    x = np.arange(nx)
    row_coords, col_coords = np.meshgrid(y, x, indexing="ij")  # row-major

    lon, lat = rasterio.transform.xy(
        transform, row_coords, col_coords, offset="center", grid=True
    )
    lon = np.array(lon).reshape(elevation_data.shape)
    lat = np.array(lat).reshape(elevation_data.shape)
    return lon, lat


def _create_contourf_levels(elevation_data: np.ndarray, interval: float) -> np.ndarray:
    """
    Computes the elevation contour levels aligned with the interval,
    starting slightly below the minimum and up to above the maximum.
    """
    min_elev = np.floor(np.min(elevation_data) / interval) * interval
    max_elev = np.ceil(np.max(elevation_data) / interval) * interval
    levels = np.arange(min_elev, max_elev + interval, interval)
    return levels


def _extract_level_polygons(cs) -> List[Tuple[float, List[Polygon]]]:
    """
    Robustly extracts filled‑contour polygons for every level in *cs*.

    We switch from ``cs.allsegs`` (boundaries only) to
    ``cs.collections[i].get_paths()``, which already encodes the
    *filled* region between ``levels[i]`` and ``levels[i+1]`` and
    therefore handles open contours along the map border without us
    guessing how to close them.

    Each Path may contain
        - the exterior ring  (first array from ``to_polygons()``)
        - zero or more interior rings (the remaining arrays)

    Returns
    -------
    list[tuple[float, list[Polygon]]]
        A tuple per level: (contour value, list of valid Polygons)
    """
    level_polys: list[tuple[float, list[Polygon]]] = []

    # ---------- Preferred method -------------------------------------------------
    if hasattr(cs, "collections"):
        for i, collection in enumerate(cs.collections):
            level = cs.levels[i]
            polys: list[Polygon] = []

            for path in collection.get_paths():
                try:
                    poly_arrays = path.to_polygons(closed_only=False)
                    if not poly_arrays:
                        continue
                    shell = poly_arrays[0]
                    holes = poly_arrays[1:] if len(poly_arrays) > 1 else None
                    poly = Polygon(shell, holes)

                    poly = clean_geometry(poly)
                    if poly:
                        polys.append(poly)

                except Exception as exc:
                    logger.warning(f"Skipping malformed path at level {level}: {exc}")

            level_polys.append((level, polys))
        return level_polys

    # ---------- Fallback method ---------------------------------------------------
    else:
        logger.warning(
            "ContourSet has no `.collections`; falling back to `.allsegs` extraction."
        )
        for i, segs in enumerate(cs.allsegs):
            level = cs.levels[i]
            polys: list[Polygon] = []

            for seg in segs:
                # Ensure the segment is closed
                if not np.allclose(seg[0], seg[-1]):
                    seg = np.vstack([seg, seg[0]])
                poly = Polygon(seg)

                poly = clean_geometry(poly)
                if poly:
                    polys.append(poly)

            level_polys.append((level, polys))
        return level_polys


def _flatten_polygons(geoms: List[Polygon]) -> List[Polygon]:
    """
    Flattens a list of geometries, extracting only the Polygon types.
    Handles MultiPolygons by extracting their constituent Polygons.
    Args:
        geoms (list): List of geometries (Polygon or MultiPolygon).
    Returns:
        list: Flattened list of Polygons.
    """
    flat = []
    for geom in geoms:
        if geom.geom_type == "Polygon":
            flat.append(geom)
        elif geom.geom_type == "MultiPolygon":
            flat.extend(g for g in geom.geoms if g.geom_type == "Polygon")
    return flat


def _force_multipolygon(geom):
    """
    Ensures the geometry is a MultiPolygon.
    If it's a Polygon, it wraps it in a MultiPolygon.
    If it's a GeometryCollection, it extracts the Polygons.
    If it's empty or not a Polygon, returns an empty MultiPolygon.
    Args:
        geom (BaseGeometry): The geometry to check.
    Returns:
        MultiPolygon: A MultiPolygon geometry."""

    if isinstance(geom, (Polygon, MultiPolygon)):
        return MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
    elif isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        return MultiPolygon(polys)
    return MultiPolygon()


def _compute_layer_bands(
    level_polys: List[Tuple[float, List[Polygon]]],
    transform: rasterio.Affine,
) -> List[dict]:
    """
    Builds a stack of cumulative contour *bands*, where each band supports
    all the layers above it (for physical stacking).

    Strategy
    --------
    1. Iterate from high → low.
    2. At each step:
       - merge current level polygons
       - union with the union of higher levels so far
       - store the result as the full base of this slice
    3. Reverse before returning to restore low → high order.
    """

    contour_layers: list[dict] = []
    cumulative = None  # union of this and all higher-level layers

    for level, polys in reversed(level_polys):
        if not polys:
            continue

        current = clean_geometry(unary_union(_flatten_polygons(polys)))
        if current is None:
            continue

        # Expand only by area: if current is fully inside, union won't change shape
        cumulative = current if cumulative is None else cumulative.union(current)
        if cumulative.is_empty:
            continue

        band = orient(_force_multipolygon(cumulative))
        contour_layers.append(
            {"elevation": float(level), "geometry": mapping(band), "closed": True}
        )

    contour_layers.reverse()
    return contour_layers


def _grid_convergence_angle_from_geometry(projected_geoms: list) -> float:
    """
    Computes the angle (in degrees) to rotate the projected geometries
    so that the bounding box aligns with the Cartesian axes.
    Uses the orientation of the minimum rotated rectangle of the union.
    Args:
        projected_geoms (list): List of projected geometries.
    Returns:
        float: Angle in degrees to rotate the geometries.
    """
    if not projected_geoms:
        return 0.0

    unioned = unary_union(projected_geoms)
    if unioned.is_empty:
        return 0.0

    min_rect = unioned.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)

    # Find the longest edge and compute its angle relative to the X-axis
    max_len = 0.0
    angle_deg = 0.0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.hypot(dx, dy)
        if length > max_len:
            max_len = length
            angle_deg = math.degrees(math.atan2(dy, dx))

    return angle_deg


def _plot_contour_layers(
    contour_layers: List[dict], raw_xlim, raw_ylim, debug_image_path: str
) -> None:
    """
    Plots all the final contour layer geometries to a debug image.
    Args:
        contour_layers (list): List of contour layers with geometries.
        raw_xlim (tuple): X-axis limits for the plot.
        raw_ylim (tuple): Y-axis limits for the plot.
        debug_image_path (str): Path to save the debug image.
    Returns:
        None
    """
    fig, ax = plt.subplots()
    for layer in contour_layers:
        band_geom = shape(layer["geometry"])
        if band_geom.geom_type == "Polygon":
            x, y = band_geom.exterior.xy
            ax.plot(x, y, linewidth=0.5)
        elif band_geom.geom_type == "LineString":
            x, y = band_geom.xy
            ax.plot(x, y, linewidth=0.5)
    ax.set_title("Closed Contours")
    ax.set_xlim(raw_xlim)
    ax.set_ylim(raw_ylim)
    ax.set_aspect("auto")
    plt.savefig(os.path.join(debug_image_path, "closed_contours.png"))
    plt.close(fig)


def generate_contours(
    elevation_data: np.ndarray,
    transform: rasterio.Affine,
    interval: float,
    simplify: float = 0.0,
    debug_image_path: str = DEBUG_IMAGE_PATH,
    center: tuple[float, float] = (0, 0),
    scale: float = 1.0,
    bounds: tuple[float, float, float, float] = None,
) -> List[dict]:
    """Generates stacked contour bands from elevation data.

    Args:
        elevation_data (ndarray): 2D elevation array.
        transform (Affine): Affine transform for spatial coordinates.
        interval (float): Height difference between layers.
        simplify (float): Optional simplification factor (not used here).
        debug_image_path (str): Path for saving debug images.
        center (tuple): Center coordinate for UTM zone.
        scale (float): Not currently used.
        bounds (tuple): Optional bounds for clipping (not used).

    Returns:
        list[dict]: Contour band geometries with elevation.
    """
    logger.debug("generate contours called")

    lon, lat = _prepare_meshgrid(elevation_data, transform)
    levels = _create_contourf_levels(elevation_data, interval)

    fig, ax = plt.subplots()
    cs = ax.contourf(lon, lat, elevation_data, levels=levels)

    if debug_image_path:
        ax.set_title("Generated Contours")
        raw_xlim = ax.get_xlim()
        raw_ylim = ax.get_ylim()
        plt.savefig(os.path.join(debug_image_path, "contours.png"))
        plt.close(fig)

    level_polys = _extract_level_polygons(cs)
    contour_layers = _compute_layer_bands(level_polys, transform)

    if DEBUG:
        os.makedirs(DEBUG_IMAGE_PATH, exist_ok=True)
        for layer in contour_layers:
            band_geom = shape(layer["geometry"])
            save_debug_contour_polygon(band_geom, layer["elevation"], "contour_polygon")

    _plot_contour_layers(contour_layers, raw_xlim, raw_ylim, debug_image_path)

    logger.debug(f"Generated {len(contour_layers)} contour layers")

    return contour_layers


def project_geometry(
    contours: List[dict], center_lon: float, center_lat: float
) -> List[dict]:
    """
    Projects a list of contour geometries from WGS84 to a local UTM zone based on the center longitude.
    Returns the list with updated projected geometries.
    Also saves a debug image of all projected geometries.
    """
    # Determine UTM zone
    zone_number = int((center_lon + 180) / 6) + 1
    is_northern = center_lat >= 0
    epsg_code = f"326{zone_number:02d}" if is_northern else f"327{zone_number:02d}"

    # Set up projection from WGS84 to UTM
    proj = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    # First, project all geometries without rotation, collect them for union
    projected_geoms = []
    for contour in contours:
        try:
            geom = shape(contour["geometry"])
            projected_geom = transform(proj.transform, geom)
            projected_geoms.append((contour, projected_geom))
        except Exception as e:
            logger.warning(
                f"Skipping unprojectable contour at elevation {contour.get('elevation')}: {e}"
            )
            continue

    # Compute common centroid for rotation
    all_shapes = [g for _, g in projected_geoms]
    if not all_shapes:
        return []
    merged = unary_union(all_shapes)
    center = merged.centroid
    rot_angle = _grid_convergence_angle_from_geometry(all_shapes)

    projected_contours = []
    fig, ax = plt.subplots()

    for contour, geom in projected_geoms:
        rotated_geom = shapely.affinity.rotate(
            geom, -rot_angle - 90, origin=center
        )  # adding 90 because everything got rotated by 90 deg cw, and i can't find the bug
        contour["geometry"] = mapping(rotated_geom)
        projected_contours.append(contour)

        # Plot for debug
        if rotated_geom.geom_type == "Polygon":
            x, y = rotated_geom.exterior.xy
            ax.plot(x, y, linewidth=0.5)
        elif rotated_geom.geom_type == "LineString":
            x, y = rotated_geom.xy
            ax.plot(x, y, linewidth=0.5)

    ax.set_title("Projected Contours")
    plt.savefig(os.path.join(DEBUG_IMAGE_PATH, "projected_contours.png"))

    return projected_contours


def scale_and_center_contours_to_substrate(
    contours: List[dict],
    substrate_size_mm: float,
    utm_bounds: Tuple[float, float, float, float],
) -> List[dict]:
    """
    Scales and centers projected contour geometries to fit within a bounding box derived from the given UTM bounds.
    Returns a new list of updated contours.
    """
    substrate_m = substrate_size_mm / 1000.0

    minx, miny, maxx, maxy = utm_bounds
    width = maxx - minx
    height = maxy - miny
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # Determine uniform scale based on the larger dimension
    scale_factor = substrate_m / max(width, height)

    elevation_groups = defaultdict(list)
    for contour in contours:
        elevation = contour["elevation"]
        geom = shape(contour["geometry"])
        elevation_groups[elevation].append(geom)

    updated = []
    for elevation, group_geoms in elevation_groups.items():
        flat_geoms = []
        for g in group_geoms:
            if g.geom_type == "Polygon":
                flat_geoms.append(g)
            elif g.geom_type == "MultiPolygon":
                flat_geoms.extend(g.geoms)
        union_geom = MultiPolygon(flat_geoms)
        moved = shapely.affinity.translate(union_geom, xoff=-center_x, yoff=-center_y)
        scaled = shapely.affinity.scale(
            moved, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
        )
        updated.append(
            {"elevation": elevation, "geometry": mapping(scaled), "closed": True}
        )

    all_scaled_bounds = [shape(c["geometry"]).bounds for c in updated]
    min_x = min(b[0] for b in all_scaled_bounds)
    max_x = max(b[2] for b in all_scaled_bounds)
    min_y = min(b[1] for b in all_scaled_bounds)
    max_y = max(b[3] for b in all_scaled_bounds)

    logger.debug(
        f"Overall scaled extent: {max_x - min_x:.4f} m × {max_y - min_y:.4f} m"
    )
    return updated
