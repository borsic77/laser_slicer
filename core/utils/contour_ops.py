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
from shapely.geometry import Polygon, mapping, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .geometry_ops import _flatten_polygons, _force_multipolygon, clean_geometry_strict

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
DEBUG = settings.DEBUG
if DEBUG:
    os.makedirs(DEBUG_IMAGE_PATH, exist_ok=True)


def save_debug_contour_polygon(polygon, level: float, filename: str) -> None:
    """Persist a debug PNG of a contour polygon.

    Args:
        polygon: The polygon to plot.
        level: Elevation level of the polygon.
        filename: Base name of the output file.
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


def _prepare_meshgrid(
    elevation_data: np.ndarray, transform: rasterio.Affine
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate longitude and latitude grids for plotting.

    Args:
        elevation_data: Elevation values as a 2D array.
        transform: Affine transform of the raster.

    Returns:
        Two arrays representing longitude and latitude respectively.
    """
    ny, nx = elevation_data.shape
    y = np.arange(ny)
    x = np.arange(nx)
    row_coords, col_coords = np.meshgrid(y, x, indexing="ij")
    lon, lat = rasterio.transform.xy(
        transform, row_coords, col_coords, offset="center", grid=True
    )
    lon = np.array(lon).reshape(elevation_data.shape)
    lat = np.array(lat).reshape(elevation_data.shape)
    return lon, lat


def _create_contourf_levels(
    elevation_data: np.ndarray,
    interval: float,
    fixed_elevation: float | None = None,
    tol: float = 1e-3,
    margin: float = 30.0,
    num_layers: int | None = None,
) -> np.ndarray:
    """Compute the contour levels for a DEM.

    Args:
        elevation_data: Raw elevation array.
        interval: Desired contour interval in metres.
        fixed_elevation: Optional fixed elevation to include.
        tol: Tolerance when building the level range.
        margin: Unused legacy parameter.
        num_layers: Generate this many evenly spaced layers instead.

    Returns:
        Array of contour break values.
    """
    min_elev = np.nanmin(elevation_data)
    max_elev = np.nanmax(elevation_data)

    if np.isnan(min_elev) or np.isnan(max_elev):
        raise ValueError(
            "Elevation data contains only NaN values. Check bounds or source data."
        )

    if num_layers is not None:
        levels = np.linspace(min_elev, max_elev, num_layers + 1).tolist()
    else:
        levels = np.arange(min_elev, max_elev + tol, interval).tolist()
    if fixed_elevation is not None:
        if min(levels) < fixed_elevation:
            v = fixed_elevation - 3.1
            if not any(abs(v - lvl) < 1e-4 for lvl in levels):
                levels.append(v)
        if max(levels) > fixed_elevation:
            v = fixed_elevation + 3.1
            if not any(abs(v - lvl) < 1e-4 for lvl in levels):
                levels.append(v)
    levels = sorted(set(round(lvl, 6) for lvl in levels))
    logger.debug(
        "Elevation data ranges from %s, %s \nContour levels boundaries are: %s",
        min_elev,
        max_elev,
        levels,
    )
    return np.array(levels)


def _extract_level_polygons(
    cs, min_area: float = 0.0
) -> List[Tuple[float, List[Polygon]]]:
    """Convert matplotlib contour output to Shapely polygons.

    Args:
        cs (QuadContourSet): A matplotlib ``QuadContourSet`` returned by ``contourf``.
        min_area (float): Minimum area to keep (in source units, likely deg²).

    Returns:
        List[Tuple[float, List[Polygon]]]: List of ``(level, polygons)`` tuples.
    """
    level_polys: list[tuple[float, list[Polygon]]] = []
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
                    # Early area check before expensive cleaning
                    if min_area > 0 and poly.area < min_area:
                        continue

                    poly = clean_geometry_strict(poly)
                    if poly:
                        polys.append(poly)
                except Exception as exc:  # pragma: no cover - log and skip
                    logger.warning(
                        "Skipping malformed path at level %s: %s", level, exc
                    )
            level_polys.append((level, polys))
        return level_polys
    logger.warning(
        "ContourSet has no `.collections`; falling back to `.allsegs` extraction."
    )
    for i, segs in enumerate(cs.allsegs):
        level = cs.levels[i]
        polys: list[Polygon] = []
        for seg in segs:
            if len(seg) < 3:
                continue
            if not np.allclose(seg[0], seg[-1]):
                seg = np.vstack([seg, seg[0]])
            poly = Polygon(seg)
            # Early area check
            if min_area > 0 and poly.area < min_area:
                continue

            poly = clean_geometry_strict(poly)
            if poly:
                polys.append(poly)
        level_polys.append((level, polys))
    return level_polys


def _plot_contour_layers(
    contour_layers: List[dict], raw_xlim, raw_ylim, debug_image_path: str
) -> None:
    """Plot contour layers for debugging purposes.

    Args:
        contour_layers: Sequence of contour features.
        raw_xlim: Original X limits from Matplotlib.
        raw_ylim: Original Y limits from Matplotlib.
        debug_image_path: Directory to write ``closed_contours.png``.
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


def _compute_layer_bands(
    level_polys: List[Tuple[float, List[Polygon]]], transform
) -> List[dict]:
    """Build a cumulative stack of closed contour bands.

    Args:
        level_polys: Output from :func:`_extract_level_polygons`.
        transform: Affine transform of the DEM.

    Returns:
        List of contour features sorted from lowest to highest.
    """
    contour_layers: list[dict] = []
    cumulative = None
    for level, polys in reversed(level_polys):
        if not polys:
            continue
        current = clean_geometry_strict(unary_union(_flatten_polygons(polys)))
        if current is None:
            logger.warning("Layer %s produced no valid geometry after cleaning.", level)
            continue
        cumulative = current if cumulative is None else cumulative.union(current)
        if cumulative.is_empty:
            continue
        band = orient(_force_multipolygon(cumulative))
        contour_layers.append(
            {"elevation": float(level), "geometry": mapping(band), "closed": True}
        )
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
    scale: float = 1.0,
    bounds: tuple[float, float, float, float] | None = None,
    fixed_elevation: float | None = None,
    num_layers: int | None = None,
    water_polygon: Polygon | None = None,
    resolution_scale: float = 1.0,
    min_area_deg2: float = 1e-10,
    dem_smoothing: float = 0.0,
) -> List[dict]:
    """Generate contour bands from elevation data.

    Args:
        masked_elevation_data: Masked DEM used for level creation.
        elevation_data: Raw DEM values.
        transform: Affine transform describing ``elevation_data``.
        interval: Contour interval in metres.
        simplify: Simplification tolerance in metres.
        debug_image_path: Directory for debug images.
        center: Center coordinates of the area of interest.
        scale: Unused scaling factor kept for backward compatibility.
        bounds: Bounding box of the area of interest in WGS84.
        fixed_elevation: Elevation at which a water body should be inserted.
        num_layers: If given, override ``interval`` and generate that many layers.
        water_polygon: Optional polygon of a water body to carve out.
        resolution_scale: Downsample factor (0.0 < x <= 1.0).
        min_area_deg2: Minimum polygon area in degrees squared to keep.
        dem_smoothing: Gaussian blur sigma for DEM pre-processing (0.0 = off).

    Returns:
        A list of contour feature dictionaries sorted from bottom to top.
    """
    logger.debug(
        "generate contours called, fixed_elevation: %s, res_scale: %s",
        fixed_elevation,
        resolution_scale,
    )

    # Downsample if requested
    if resolution_scale < 1.0 and resolution_scale > 0:
        step = int(1 / resolution_scale)
        if step > 1:
            logger.info("Downsampling elevation data by factor %d", step)
            elevation_data = elevation_data[::step, ::step]
            masked_elevation_data = masked_elevation_data[::step, ::step]
            # Update transform: pixel size increases by factor 'step'
            transform = transform * transform.scale(step, step)

    # Apply light Gaussian smoothing to DEM to remove pixel stair-stepping (jagged edges)
    if dem_smoothing > 0:
        logger.info("Applying Gaussian smoothing to DEM (sigma=%.2f)", dem_smoothing)
        elevation_data = scipy.ndimage.gaussian_filter(
            elevation_data, sigma=dem_smoothing
        )
        masked_elevation_data = scipy.ndimage.gaussian_filter(
            masked_elevation_data, sigma=dem_smoothing
        )

    lon, lat = _prepare_meshgrid(elevation_data, transform)
    levels = _create_contourf_levels(
        masked_elevation_data, interval, fixed_elevation, num_layers=num_layers
    )
    fig, ax = plt.subplots()
    cs = ax.contourf(lon, lat, elevation_data, levels=levels)
    if debug_image_path:
        ax.set_title("Generated Contours")
        raw_xlim = ax.get_xlim()
        raw_ylim = ax.get_ylim()
        plt.savefig(os.path.join(debug_image_path, "contours.png"))

    # Pass min_area for early filtering
    level_polys = _extract_level_polygons(cs, min_area=min_area_deg2)
    plt.close(fig)

    if water_polygon is not None and fixed_elevation is not None:
        water_band = orient(_force_multipolygon(water_polygon))
        new_level_polys = []
        inserted = False
        for i, (level, polys) in enumerate(level_polys):
            if level < fixed_elevation - 3:
                new_level_polys.append((level, polys))
            elif abs(level - fixed_elevation) <= 3:
                continue
            elif not inserted and level > fixed_elevation + 3:
                new_level_polys.append((fixed_elevation, [water_band]))
                inserted = True
                cleaned_water = clean_geometry_strict(water_band)
                # Buffer by approx 5 meters (0.00005 deg) to ensure water fits inside the hole
                # Previous value of 0.01 was ~1km, which deleted small lakes!
                robust_water = (
                    cleaned_water.buffer(-0.00005)
                    if cleaned_water is not None
                    else None
                )
                robust_water = (
                    clean_geometry_strict(robust_water)
                    if robust_water is not None
                    else None
                )
                above_bands = level_polys[i:]
                for above_level, above_polys in above_bands:
                    cleaned_land = [
                        clean_geometry_strict(p)
                        for p in above_polys
                        if p and not p.is_empty
                    ]
                    cleaned_land = [
                        p for p in cleaned_land if p is not None and not p.is_empty
                    ]
                    if (
                        cleaned_land
                        and robust_water is not None
                        and not robust_water.is_empty
                    ):
                        carved = [p.difference(robust_water) for p in cleaned_land]
                        carved_clean = [
                            clean_geometry_strict(p)
                            for p in carved
                            if p and not p.is_empty and clean_geometry_strict(p)
                        ]
                        new_level_polys.append((above_level, carved_clean))
                    else:
                        new_level_polys.append((above_level, cleaned_land))
                break
            else:
                new_level_polys.append((level, polys))
        if not inserted:
            new_level_polys.append((fixed_elevation, [water_band]))
        level_polys = new_level_polys

    contour_layers = _compute_layer_bands(level_polys, transform)
    if DEBUG:
        os.makedirs(DEBUG_IMAGE_PATH, exist_ok=True)
        for layer in contour_layers:
            band_geom = shape(layer["geometry"])
            save_debug_contour_polygon(band_geom, layer["elevation"], "contour_polygon")
    _plot_contour_layers(contour_layers, raw_xlim, raw_ylim, debug_image_path)
    logger.debug("Generated %d contour layers", len(contour_layers))

    filtered = []
    for layer in contour_layers:
        geom = shape(layer["geometry"])
        if not geom.is_empty and geom.area > min_area_deg2:
            filtered.append(layer)

    logger.debug("Returning %d contours after area filtering", len(filtered))
    return filtered
