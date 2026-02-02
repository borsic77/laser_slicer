"""Functions for generating contour polygons from DEM data."""

import logging
import os
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import scipy.ndimage
from django.conf import settings
from shapely.geometry import LinearRing, Polygon, mapping, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from skimage import measure

from .geometry_ops import _flatten_polygons, _force_multipolygon, clean_geometry_strict

# Minimal matplotlib usage for debug plotting only
matplotlib.use("Agg")
logger = logging.getLogger(__name__)

DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
DEBUG = settings.DEBUG
if DEBUG:
    os.makedirs(DEBUG_IMAGE_PATH, exist_ok=True)


def save_debug_contour_polygon(polygon, level: float, filename: str) -> None:
    """Persist a debug PNG of a contour polygon."""
    if not DEBUG:
        return
    fig, ax = plt.subplots()
    if polygon.is_empty or not polygon.is_valid:
        plt.close(fig)
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


def _contours_to_polygons(
    elevation_data: np.ndarray,
    level: float,
    transform: rasterio.Affine,
) -> List[Polygon]:
    """Generate polygons for a specific elevation level using skimage marching squares.

    Args:
        elevation_data: 2D array of elevation values.
        level: Elevation value to contour at.
        transform: Affine transform to convert image coords to geospatial coords.

    Returns:
        List of closed Polygons from the isolines.
        Note: find_contours returns lines. We convert closed lines to Polygons.
        This represents the "isoline" at exactly `level`.
        To get filled "everything above", we will rely on layer stacking/union or
        assume the loop encloses a peak (for simple hills).
        For complex topology, the "cumulative union" strategy in `_compute_layer_bands`
        handles the filling logic by stacking them.
    """
    # Strategy: Pad the array with a low value so that all valid terrain
    # forms closed "islands" inside the padded array.
    # This methodology forces find_contours to close the loops at boundaries.

    pad_width = 1
    # Use a value lower than any possible data, e.g. -infinity or just deep void
    # -99999.0 is safe for terrestrial/bathymetry data.
    padded_data = np.pad(
        elevation_data, pad_width=pad_width, mode="constant", constant_values=-99999.0
    )

    # contours is list of (row, column)
    # fully_connected="high" matches matplotlib's default connectivity
    contours = measure.find_contours(padded_data, level, fully_connected="high")

    polys = []
    for contour in contours:
        # contour is (row, col) in PADDED space.
        # Shift back by pad_width to get original image coordinates.
        rows = contour[:, 0] - pad_width
        cols = contour[:, 1] - pad_width

        # rasterio transform: (row, col) -> (x, y)
        # rasterio expects (row, col) to output (x, y)
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

        # Combine to standard (x, y) coordinates
        coords = list(zip(xs, ys))

        # Filter small artifacts (less than 3 points)
        if len(coords) < 3:
            continue

        # Ensure closure
        if coords[0] != coords[-1]:
            # Close it
            coords.append(coords[0])

        try:
            # Create a LinearRing first to check validity
            ring = LinearRing(coords)
            if ring.is_valid:
                poly = Polygon(ring)
                # Cleaning is done later, but minimal clean here is good
                if poly.is_valid:
                    polys.append(poly)
            else:
                # Try cleaning self-intersections via buffer(0)
                poly = Polygon(coords).buffer(0)
                if poly and not poly.is_empty:
                    if poly.geom_type == "MultiPolygon":
                        polys.extend(poly.geoms)
                    else:
                        polys.append(poly)

        except Exception as e:
            logger.debug(f"Failed to create polygon from contour: {e}")
            continue

    return polys


def _create_levels(
    min_elev: float,
    max_elev: float,
    interval: float,
    fixed_elevation: float | None = None,
    num_layers: int | None = None,
) -> List[float]:
    """Compute contour levels."""
    if np.isnan(min_elev) or np.isnan(max_elev):
        raise ValueError("Elevation data contains NaN.")

    if num_layers is not None:
        levels = np.linspace(min_elev, max_elev, num_layers + 1).tolist()
        if min_elev < 0 < max_elev:
            levels.append(0.0)
    else:
        start = np.floor(min_elev / interval) * interval
        levels = np.arange(start, max_elev + 1e-3, interval).tolist()
        if min_elev < 0 < max_elev:
            levels.append(0.0)

    if fixed_elevation is not None:
        if min(levels) < fixed_elevation < max(levels):
            levels.append(fixed_elevation)

    levels = sorted(set(round(lvl, 6) for lvl in levels))
    levels = [l for l in levels if l >= min_elev and l <= max_elev]
    return levels


def _compute_layer_bands(
    level_polys: List[Tuple[float, List[Polygon]]],
    transform: rasterio.Affine,
    simplify_tolerance: float = 0.0,
) -> List[dict]:
    """Build a cumulative stack of contour bands.

    This replicates the original logic:
    - Sort levels High to Low.
    - Compute Union of all polygons at Level X.
    - Accumulate: Band(X) = Poly(X) U Band(X+1).
    - This ensures that if we have a peak at 50m, the 40m layer ALSO covers the 50m peak.
    """
    contour_layers: list[dict] = []

    # Sort High to Low for accumulation
    sorted_levels = sorted(level_polys, key=lambda x: x[0], reverse=True)

    cumulative = None

    for level, polys in sorted_levels:
        if not polys:
            continue

        # 1. Union all polygons at this specific level (e.g. multiple islands)
        current_level_geom = unary_union(polys)
        current_level_geom = clean_geometry_strict(current_level_geom)

        if current_level_geom is None or current_level_geom.is_empty:
            continue

        # 2. Add to cumulative stack (Everything above this level)
        if cumulative is None:
            cumulative = current_level_geom
        else:
            cumulative = cumulative.union(current_level_geom)

        cumulative = clean_geometry_strict(cumulative)

        if cumulative and not cumulative.is_empty:
            # Apply simplification if requested
            if simplify_tolerance > 0:
                cumulative = cumulative.simplify(
                    simplify_tolerance, preserve_topology=True
                )

            band = orient(_force_multipolygon(cumulative))
            pass

            contour_layers.append(
                {"elevation": float(level), "geometry": mapping(band), "closed": True}
            )

    # Reverse back to Low -> High for final output
    contour_layers.reverse()
    return contour_layers


def generate_contours(
    masked_elevation_data: np.ndarray,
    elevation_data: np.ndarray,
    transform: rasterio.Affine,
    interval: float,
    simplify: float = 0.0,
    debug_image_path: str = DEBUG_IMAGE_PATH,
    center: tuple[float, float] = (0, 0),
    bounds: tuple[float, float, float, float] | None = None,
    fixed_elevation: float | None = None,
    num_layers: int | None = None,
    water_polygon: Polygon | None = None,
    resolution_scale: float = 1.0,
    min_area_deg2: float = 1e-10,
    dem_smoothing: float = 0.0,
) -> List[dict]:
    """Generate contours using skimage marching squares (smooth)."""

    logger.debug(f"generate_contours (skimage) called. Smoothing: {dem_smoothing}")

    # 1. Smoothing
    if dem_smoothing > 0:
        elevation_data = scipy.ndimage.gaussian_filter(
            elevation_data, sigma=dem_smoothing
        )

    # 2. Prepare Levels
    min_val = np.nanmin(elevation_data)
    max_val = np.nanmax(elevation_data)
    levels = _create_levels(min_val, max_val, interval, fixed_elevation, num_layers)
    logger.debug(f"Levels: {levels}")

    # 3. Extract Polygons for each level
    level_polys_list = []

    for level in levels:
        # Fill NaNs with a value well below min to ensure they are "outside"
        data_safe = elevation_data.copy()
        data_safe[np.isnan(data_safe)] = -99999.0

        polys = _contours_to_polygons(data_safe, level, transform)
        if polys:
            # Filter small
            if min_area_deg2 > 0:
                polys = [p for p in polys if p.area >= min_area_deg2]
            if polys:
                level_polys_list.append((level, polys))

    # 4. Handle Water / Fixed Elevation (Simple version: Insert water logic here)
    if fixed_elevation is not None and water_polygon is not None:
        # Skipped complex water logic for this refactor verification step
        pass

    # 5. Compute Stacked Bands
    contour_layers = _compute_layer_bands(
        level_polys_list, transform, simplify_tolerance=simplify
    )

    logger.debug(f"Generated {len(contour_layers)} layers.")
    return contour_layers
