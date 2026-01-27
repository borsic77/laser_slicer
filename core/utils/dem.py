"""Utilities for manipulating digital elevation models (DEMs)."""

import logging
import math
from typing import List, Tuple

import numpy as np
import rasterio
import scipy.ndimage
from rasterio.merge import merge
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)


def clean_srtm_dem(
    dem: np.ndarray, min_valid: float = -500, max_valid: float = 9000
) -> np.ndarray:
    """Clean invalid values from a DEM.

    Args:
        dem: DEM values as a 2D array.
        min_valid: Lowest acceptable elevation in metres.
        max_valid: Highest acceptable elevation in metres.

    Returns:
        ``dem`` with out-of-range values replaced by ``np.nan``.
    """

    arr = np.where((dem == -32768) | (dem < min_valid) | (dem > max_valid), np.nan, dem)
    return arr


def robust_local_outlier_mask(
    arr: np.ndarray, window: int = 5, thresh: float = 5.0
) -> np.ndarray:
    """Mask local elevation outliers.

    Args:
        arr: DEM array to clean.
        window: Size of the median filter window.
        thresh: Threshold multiplier for the MAD.

    Returns:
        ``arr`` with detected outliers replaced by ``np.nan``.
    """

    med = scipy.ndimage.median_filter(arr, size=window, mode="reflect")
    diff = np.abs(arr - med)
    mad = scipy.ndimage.median_filter(diff, size=window, mode="reflect")
    mask = diff > thresh * (mad + 1e-6)
    return np.where(mask, np.nan, arr)


def fill_nans_in_dem(arr: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Fill missing elevation values iteratively.

    Args:
        arr: DEM array containing ``np.nan`` gaps.
        max_iter: Maximum number of interpolation passes.

    Returns:
        The array with NaN values filled by nearest-neighbour interpolation.
    """

    arr = arr.copy()
    for _ in range(max_iter):
        if not np.isnan(arr).any():
            break
        mask = np.isnan(arr)
        filled = scipy.ndimage.generic_filter(arr, np.nanmean, size=5, mode="nearest")
        arr[mask] = filled[mask]
    return arr


def round_affine(
    transform: rasterio.Affine, precision: float = 1e-9
) -> rasterio.Affine:
    """Round an affine transform for stable comparisons.

    Args:
        transform: Affine transform to round.
        precision: Decimal precision. For WGS84 (degrees), 1e-4 is approx 11 meters,
                   which is too coarse for 2m pixels (approx 2e-5).
                   Use 1e-9 (approx 0.1mm) by default.

    Returns:
        A transform with each component rounded to ``precision``.
    """

    return rasterio.Affine(
        *(round(val, int(-math.log10(precision))) for val in transform)
    )


def sample_elevation(lat: float, lon: float, dem_path: str) -> float:
    """Read a single elevation value from a DEM file.

    Args:
        lat: Latitude in WGS84.
        lon: Longitude in WGS84.
        dem_path: Path to a (possibly gzipped) DEM raster.

    Returns:
        The elevation in metres at the specified coordinates.
    """

    with rasterio.open(f"/vsigzip/{dem_path}") as src:
        row, col = src.index(lon, lat)
        arr = src.read(1)
        logger.debug(
            "Sampled elevation at (%s, %s) -> (row: %s, col: %s), value: %s",
            lat,
            lon,
            row,
            col,
            arr[row, col],
        )
        return float(arr[row, col])


def mosaic_and_crop(
    tif_paths: List[str],
    bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, rasterio.Affine]:
    """Merge elevation tiles and crop to a bounding box.

    Args:
        tif_paths: Paths to the tiles to merge. ``.gz`` files are opened via ``vsigzip``.
        bounds: Bounding box ``(lon_min, lat_min, lon_max, lat_max)`` in WGS84.

    Returns:
        A tuple ``(array, transform)`` of the cropped DEM and its affine transform.
    """
    if not tif_paths:
        raise ValueError("No elevation tiles provided for mosaicing.")

    src_files = [
        rasterio.open(f"/vsigzip/{p}") if p.endswith(".gz") else rasterio.open(p)
        for p in tif_paths
    ]
    mosaic, transform = merge(src_files)

    lon_min, lat_min, lon_max, lat_max = bounds
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    bounds = (lon_min, lat_min, lon_max, lat_max)

    if transform.e > 0:
        logger.warning(
            "Unexpected raster orientation (north-up). SRTM tiles usually have north-down orientation."
        )

    try:
        window = from_bounds(*bounds, transform=transform)
    except ValueError:
        logger.error(
            "Failed to compute raster window from bounds %s with transform %s",
            bounds,
            transform,
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
