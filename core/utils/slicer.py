"""
Core utilities for slicing SRTM (DEM) data into stacked contour bands for laser cutting.

Key responsibilities:
- Merging/cropping DEM tiles to area of interest
- Generating robust, closed polygons for each contour interval
- Geometric cleaning, simplification, and projection
- Scaling contours to match laser substrate dimensions
- Filtering out small or narrow features unsuitable for cutting

Most functions are pure and stateless, facilitating unit testing and background task execution.
"""

import logging
import math
import os
from collections import defaultdict
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import shapely
from django.conf import settings
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
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

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


def clean_geometry_strict(geom: BaseGeometry) -> BaseGeometry | None:
    """Aggressively attempts to produce a valid polygon.
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
    if not geom.is_valid:
        # Shapely's buffer(0) is a standard trick to clean up slight self-intersections in polygons.
        try:
            geom = geom.buffer(0)
        except Exception:
            return None
    if geom.is_empty or geom.area == 0:
        return None
    if not geom.is_valid:
        return None
    # Remove GeometryCollections with only lines/points
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        if len(polys) == 1:
            geom = polys[0]
        else:
            geom = MultiPolygon(polys)
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


def sample_elevation(lat: float, lon: float, dem_path) -> float:
    """Samples the elevation at a specific latitude and longitude from a DEM file.
    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        dem_path (str): Path to the DEM file (GeoTIFF or HGT).
    Returns:
        float: The elevation at the specified coordinates.
    """
    with rasterio.open(f"/vsigzip/{dem_path}") as src:
        row, col = src.index(lon, lat)
        arr = src.read(1)
        logger.debug(
            f"Sampled elevation at ({lat}, {lon}) -> (row: {row}, col: {col}), value: {arr[row, col]}"
        )
        return float(arr[row, col])


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

    # Ensure bounds are ordered (south < north), as rasterio expects this order for windows
    # SRTM data (GeoTIFF/HGT) can be in either north-up or south-up order. Rasterio expects bounds as (min, max), regardless of raster orientation.

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


# not used right now, delete later
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


def _create_contourf_levels(
    elevation_data: np.ndarray,
    interval: float,
    fixed_elevation: float = None,
    tol: float = 1e-3,
    margin: float = 30.0,
) -> np.ndarray:
    """Creates contour levels for filled contours based on elevation data.
    Computes the minimum and maximum elevation values, then generates
    levels at specified intervals. If a fixed elevation is provided,
    it adds the water band edges with a margin.
    Args:
        elevation_data (np.ndarray): 2D array of elevation values.
        interval (float): Height difference between contour levels.
        fixed_elevation (float, optional): If provided, adds water band edges
            at this elevation with a margin.
        tol (float): Tolerance for rounding contour levels.
        margin (float): Margin to add around the fixed elevation.
    Returns:
        np.ndarray: Array of contour levels.
    """
    min_elev = np.floor(np.min(elevation_data) / interval) * interval
    max_elev = np.ceil(np.max(elevation_data) / interval) * interval
    levels = np.arange(min_elev, max_elev + tol, interval).tolist()

    if fixed_elevation is not None:
        # Insert water band edges, with a little margin
        for v in (fixed_elevation - margin, fixed_elevation + margin):
            if not any(abs(v - lvl) < margin for lvl in levels):
                levels.append(v)
    # Sort and unique
    levels = sorted(set(round(lvl, 6) for lvl in levels))
    logger.debug("Contour levels boundaries are: %s", levels)
    return np.array(levels)


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
    # Fallback: if collections is not available, use allsegs
    # fall back to assembling closed polygons from boundary segments (less robust, but works for simple DEMs)
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

    Build a cumulative stack of closed contour bands for laser-cuttable physical models.

    For each elevation band (highest to lowest), we compute the union of this band's polygons
    with all bands above it. This ensures every slice includes the full area of all upper layers,
    guaranteeing proper physical support and alignment—critical for real-world assembly.

    We clean geometries aggressively, as SRTM contours may have small self-intersections or artefacts.


    Strategy
    --------
    1. Iterate from high → low.
    2. At each step:
       - merge current level polygons
       - union with the union of higher levels so far
       - store the result as the full base of this slice
    3. Reverse before returning to restore low → high order.
    """

    # Each physical contour must support all layers above it, so we union downward (from top to bottom) to ensure proper stacking support.
    contour_layers: list[dict] = []
    cumulative = None  # union of this and all higher-level layers

    for level, polys in reversed(level_polys):
        if not polys:
            continue

        current = clean_geometry_strict(unary_union(_flatten_polygons(polys)))
        if current is None:
            logger.warning(f"Layer {level} produced no valid geometry after cleaning.")
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
    fixed_elevation: float = None,
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
        fixed_elevation (float): If provided, starts slicing from this elevation.

    Returns:
        list[dict]: Contour band geometries with elevation.
    """
    logger.debug("generate contours called, fixed_elevation: %s", fixed_elevation)

    lon, lat = _prepare_meshgrid(elevation_data, transform)
    levels = _create_contourf_levels(elevation_data, interval, fixed_elevation)

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
    # Filter out empty or invalid geometries
    filtered = []
    for layer in contour_layers:
        geom = shape(layer["geometry"])
        if not geom.is_empty and geom.area > 1e-8:
            filtered.append(layer)
    contour_layers = filtered

    return contour_layers


def project_geometry(
    contours: List[dict],
    center_lon: float,
    center_lat: float,
    simplify_tolerance: float = 0.0,
) -> List[dict]:
    """
    Projects a list of contour geometries from WGS84 to a local UTM zone based on the center longitude.
    Returns the list with updated projected geometries.
    Also saves a debug image of all projected geometries.
    Args:
        contours (List[dict]): List of contour dictionaries with 'geometry' in WGS84.
        center_lon (float): Longitude of the center point for UTM zone determination.
        center_lat (float): Latitude of the center point for UTM zone determination.
        simplify_tolerance (float): Tolerance for simplifying geometries.
    Returns:
        List[dict]: Updated list of contours with projected geometries.
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
        # Clean geometry strictly after rotation
        cleaned_geom = clean_geometry_strict(rotated_geom)
        logger.debug(
            f"Contour at elevation {contour.get('elevation')} cleaned geometry is "
            f"{'valid' if cleaned_geom is not None else 'invalid'}"
        )
        if cleaned_geom is None:
            logger.warning(
                f"Skipping contour at elevation {contour.get('elevation')} after cleaning (invalid geometry)."
            )
            continue
        final_geom = cleaned_geom
        if simplify_tolerance > 0.0:
            logger.debug(
                f"Simplifying geometry for elevation {contour['elevation']} with tolerance {simplify_tolerance}"
            )
            final_geom = final_geom.simplify(simplify_tolerance, preserve_topology=True)
        contour["geometry"] = mapping(final_geom)
        projected_contours.append(contour)

        # Plot for debug
        if final_geom.geom_type == "Polygon":
            x, y = final_geom.exterior.xy
            ax.plot(x, y, linewidth=0.5)
        elif final_geom.geom_type == "LineString":
            x, y = final_geom.xy
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
    Args:
        contours (List[dict]): List of contour dictionaries with projected 'geometry'.
        substrate_size_mm (float): Size of the substrate in millimeters.
        utm_bounds (Tuple[float, float, float, float]): UTM bounding box (minx, miny, maxx, maxy).
    Returns:
        List[dict]: Updated list of contours with scaled and centered geometries.
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


def smooth_geometry(
    contours: List[dict],
    smoothing: int,
) -> List[dict]:
    """
    Applies optional smoothing to projected contour geometries.

    Args:
        contours (List[dict]): List of contour dictionaries with projected 'geometry'.
        smoothing (int): Buffer radius multiplier for smoothing (0 = no smoothing).

    Returns:
        List[dict]: Updated list of contours with smoothed geometries.
    """
    if smoothing <= 0:
        return contours

    smoothed_contours = []

    for contour in contours:
        geom = shape(contour["geometry"])

        if smoothing > 0:
            geom = geom.buffer(smoothing).buffer(-smoothing)
        logger.debug(
            f"Contour at elevation {contour['elevation']} smoothed with radius {smoothing}"
        )

        contour["geometry"] = mapping(geom)
        smoothed_contours.append(contour)

    return smoothed_contours


# --- Minimum area filtering for projected/scaled contours ---


def filter_small_features(
    contours: List[dict], min_area_cm2: float, min_width_mm: float = 0.0
) -> List[dict]:
    """
    Filters out polygons smaller than the specified area (in cm²)
    from the already projected and scaled contours.

    Args:
        contours (List[dict]): List of contour dictionaries with projected and scaled 'geometry'.
        min_area_cm2 (float): Minimum polygon area in cm² to keep.
        min_width_mm (float): Minimum width in mm for filtering.

    Returns:
        List[dict]: Filtered list of contours.
    """
    min_area_m2 = min_area_cm2 / 1e4 if min_area_cm2 > 0 else 0.0  # convert to m²
    min_width_m = min_width_mm / 1000.0 if min_width_mm > 0 else 0.0

    filtered = []

    for contour in contours:
        geom = shape(contour["geometry"])

        # --- Minimum width filtering (buffer-in/out trick) ---
        if min_width_m > 0:
            try:
                geom = geom.buffer(-min_width_m / 2).buffer(min_width_m / 2)
                geom = clean_geometry_strict(geom)
                if geom is None or geom.is_empty:
                    logger.debug(
                        f"Contour @ {contour.get('elevation')} filtered away (min width)"
                    )
                    continue
            except Exception as e:
                logger.warning(f"Buffer trick failed for min width filtering: {e}")
                continue

        # --- Minimum area filtering ---
        parts = [g for g in _flatten_polygons([geom]) if g.area >= min_area_m2]
        if not parts:
            continue
        geom = MultiPolygon(parts)

        contour["geometry"] = mapping(geom)
        filtered.append(contour)

    return filtered
