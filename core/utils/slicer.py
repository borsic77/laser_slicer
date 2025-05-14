import logging
import os
from math import cos, radians
from typing import List, Tuple

import elevation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from django.conf import settings
from rasterio.merge import merge
from rasterio.windows import from_bounds
from shapely.geometry import Polygon

# Use headless matplotlib
matplotlib.use("Agg")

SRTM_CACHE_DIR = settings.SRTM_CACHE_DIR
DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
logger = logging.getLogger(__name__)


def download_srtm_tiles_for_bounds(
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    """
    Downloads all SRTM tiles that intersect the given bounding box.
    Returns a list of file paths to the GeoTIFFs.
    """
    os.makedirs(SRTM_CACHE_DIR, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bounds

    lat_range = range(int(lat_min), int(lat_max) + 1)
    lon_range = range(int(lon_min), int(lon_max) + 1)

    paths = []

    for lat in lat_range:
        for lon in lon_range:
            tile_bounds = (lon, lat, lon + 1, lat + 1)
            tif_path = os.path.join(SRTM_CACHE_DIR, f"srtm_{lat}_{lon}.tif")
            if not os.path.exists(tif_path):
                elevation.clip(bounds=tile_bounds, output=tif_path)
            paths.append(tif_path)

    return paths


def mosaic_and_crop(
    tif_paths: List[str], bounds: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Merges multiple SRTM tiles and clips them to the specified bounds.
    Returns a single numpy array and the affine transform.
    """
    src_files = [rasterio.open(p) for p in tif_paths]

    # Merge to a single raster
    mosaic, transform = merge(src_files)

    # Clip using bounds
    window = from_bounds(*bounds, transform=transform)
    row_off = int(window.row_off)
    row_end = row_off + int(window.height)
    col_off = int(window.col_off)
    col_end = col_off + int(window.width)

    clipped = mosaic[:, row_off:row_end, col_off:col_end]

    cropped_transform = rasterio.windows.transform(window, transform)

    # Close opened files
    for src in src_files:
        src.close()

    return clipped[0], cropped_transform


from shapely.geometry import box


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
    """
    Generate simplified contour polygons from elevation data.
    Optionally saves a debug image of the contours.
    """
    logger.debug("generate contours called")
    ny, nx = elevation_data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    x_coords, y_coords = np.meshgrid(x, y)

    lon, lat = rasterio.transform.xy(
        transform, y_coords, x_coords, offset="center", grid=True
    )
    lon = np.array(lon).reshape(elevation_data.shape)
    lat = np.array(lat).reshape(elevation_data.shape)

    logger.debug(
        f"min/max elevation: {np.nanmin(elevation_data)} / {np.nanmax(elevation_data)}"
    )
    logger.debug(
        f"shapes: lon {lon.shape}, lat {lat.shape}, data {elevation_data.shape}"
    )

    fig, ax = plt.subplots()
    cs = ax.contour(
        lon,
        lat,
        elevation_data,
        levels=np.arange(np.min(elevation_data), np.max(elevation_data), interval),
    )

    if debug_image_path:
        ax.set_title("Generated Contours")
        plt.savefig(debug_image_path)

    plt.close(fig)

    contour_layers = []
    bbox = None
    if bounds is not None:
        bbox = box(*bounds)

    lat_correction = 1 / cos(radians(center[1]))

    for level, seglist in zip(cs.levels, cs.allsegs):
        layer_polys = []
        for seg in seglist:
            if len(seg) >= 3:
                poly = Polygon(seg)
                if simplify > 0.0:
                    poly = poly.simplify(simplify, preserve_topology=True)

                if not poly.is_valid:
                    poly = poly.buffer(0)

                if bbox is not None and poly.is_valid:
                    poly = poly.intersection(bbox)

                if poly.is_valid and not poly.is_empty:
                    # Geometry-type-safe handling for Polygon/MultiPolygon
                    if poly.geom_type == "Polygon":
                        coords = [poly.exterior.coords]
                    elif poly.geom_type == "MultiPolygon":
                        coords = [p.exterior.coords for p in poly.geoms]
                    else:
                        continue  # skip non-polygonal geometries

                    for ring in coords:
                        layer_polys.append(
                            [
                                [
                                    (x - center[0]) * scale,
                                    (y - center[1]) * scale * lat_correction,
                                ]
                                for x, y in ring
                            ]
                        )
        if layer_polys:
            contour_layers.append({"elevation": level, "points": layer_polys})

    return contour_layers
