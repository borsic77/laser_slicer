"""Utilities for manipulating digital elevation models (DEMs)."""

import logging
import math
from math import floor
from pathlib import Path

import numpy as np
import rasterio
import requests
import scipy.ndimage
from django.conf import settings
from filelock import FileLock
from rasterio.merge import merge
from rasterio.windows import from_bounds

from core.utils.swiss_topo_api import download_swissalti3d_tiles

try:
    from core.utils.cache_ops import enforce_cache_limit, touch_cache_file
except ImportError:  # pragma: no cover

    def touch_cache_file(path):
        pass

    def enforce_cache_limit(cache_dir, max_bytes):
        pass


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


def merge_srtm_and_etopo(
    srtm_dem: np.ndarray,
    etopo_dem: np.ndarray,
    land_mask: np.ndarray | None = None,
    transform: rasterio.Affine | None = None,
) -> np.ndarray:
    """Merge high-res SRTM (Land) with high-res Bathymetry (Ocean) using a land mask.

    Args:
        srtm_dem: High resolution land data (SRTM).
        etopo_dem: Bathymetry data (GEBCO/ETOPO/EMODnet).
        land_mask: Optional boolean mask where True indicates Land.
                   If not provided, falls back to (srtm > 0).
        transform: Optional Affine transform (unused now, kept for compatibility/future).

    Returns:
        Combined DEM with SRTM on land and upsampled bathymetry in ocean,
        seamlessly blended at the coastline.
    """

    # Safety check
    if etopo_dem.size == 0:
        return srtm_dem
    if srtm_dem.size == 0:
        return etopo_dem

    # 1. Clean NaNs in Bathymetry before cubic spline zoom (order=3 spreads NaNs globally)
    etopo_clean = np.where(
        np.isnan(etopo_dem) | (etopo_dem <= -32768), 0.0, etopo_dem
    ).astype(np.float32)
    zoom_factors = (
        srtm_dem.shape[0] / etopo_clean.shape[0],
        srtm_dem.shape[1] / etopo_clean.shape[1],
    )
    bathy_upsampled = scipy.ndimage.zoom(etopo_clean, zoom_factors, order=3)

    if bathy_upsampled.shape != srtm_dem.shape:
        # Resize/Crop to exact match
        diff_h = bathy_upsampled.shape[0] - srtm_dem.shape[0]
        diff_w = bathy_upsampled.shape[1] - srtm_dem.shape[1]
        if diff_h > 0:
            bathy_upsampled = bathy_upsampled[:-diff_h, :]
        elif diff_h < 0:
            bathy_upsampled = np.pad(
                bathy_upsampled, ((0, -diff_h), (0, 0)), mode="edge"
            )

        if diff_w > 0:
            bathy_upsampled = bathy_upsampled[:, :-diff_w]
        elif diff_w < 0:
            bathy_upsampled = np.pad(
                bathy_upsampled, ((0, 0), (0, -diff_w)), mode="edge"
            )

    # 2. Land Masking
    if land_mask is not None:
        # Use provided mask (e.g. from OSM Tiles)
        is_land = land_mask.astype(bool)
    else:
        # Fallback: SRTM positive and valid
        logger.debug(
            "No land_mask provided to merge_srtm_and_etopo. Falling back to value-based masking (srtm > 0)."
        )
        is_land = (srtm_dem > 0) & (~np.isnan(srtm_dem)) & (srtm_dem != -32768)

    # 3. Distance-based Blending at Coastline
    if land_mask is not None:
        # If we have a high-quality explicit mask, we do NOT want to blend.
        # Blending would "smear" the coastline and deviate from the mask.
        # We want an exact match: Land is Land, Water is Water.
        land_weight = is_land.astype(np.float32)  # 1.0 where Land, 0.0 where Water
        sea_weight = (~is_land).astype(np.float32)  # 1.0 where Water, 0.0 where Land
    else:
        # We want SRTM to transition to 0.0 at the coast, and Bathy to do the same.
        # We'll use a 100m blend (approx 3-4 pixels at 30m).
        pixel_res = 30.0  # Approximate
        blend_dist_m = 100.0
        blend_width_pixels = blend_dist_m / pixel_res

        # Distance Transform: Distance in pixels to the nearest "Sea" (0 in is_land)
        dist_to_sea = scipy.ndimage.distance_transform_edt(is_land)
        # Blend Weight: 0.0 at sea, 1.0 deep inland (>100m)
        land_weight = np.clip(dist_to_sea / blend_width_pixels, 0, 1)

        # Similar for sea: Distance in pixels to nearest "Land"
        dist_to_land = scipy.ndimage.distance_transform_edt(~is_land)
        # Blend Weight: 0.0 at land, 1.0 deep sea (>100m)
        sea_weight = np.clip(dist_to_land / blend_width_pixels, 0, 1)

    # 4. Final Merge
    # Forced Coastline Point: 0.0m
    # Combine:
    # - Deep Land: SRTM
    # - Near Coast Land: SRTM blended with 0.0
    # - Near Coast Sea: 0.0 blended with Bathy
    # - Deep Sea: Bathy

    # Clean SRTM/Bathy local copies for merging
    srtm_clean = srtm_dem.copy()
    srtm_clean[(srtm_dem == -32768) | np.isnan(srtm_dem)] = 0.0

    bathy_clean = bathy_upsampled.copy()
    bathy_clean[np.isnan(bathy_clean)] = 0.0

    final_dem = np.zeros(srtm_dem.shape, dtype=np.float32)

    # Calculate merged values
    # Land area: Blend SRTM towards 0.0 as we approach the coast
    final_dem[is_land] = srtm_clean[is_land] * land_weight[is_land]

    # Sea area: Blend Bathy towards 0.0 as we approach the coast
    final_dem[~is_land] = bathy_clean[~is_land] * sea_weight[~is_land]

    return final_dem


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
    """Fill missing elevation values by nearest-neighbour interpolation."""
    was_masked = isinstance(arr, np.ma.MaskedArray)
    data = arr.filled(np.nan).copy() if was_masked else np.array(arr, copy=True)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    mask = np.isnan(data)
    if not mask.any() or mask.all():
        return np.ma.masked_invalid(data) if was_masked else data

    indices = scipy.ndimage.distance_transform_edt(
        mask, return_distances=False, return_indices=True
    )
    data = data[tuple(indices)]
    return np.ma.masked_invalid(data) if was_masked else data


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
        window = rasterio.windows.Window(col, row, 1, 1)
        arr = src.read(1, window=window)
        val = arr[0, 0]
        logger.debug(
            "Sampled elevation at (%s, %s) -> (row: %s, col: %s), value: %s",
            lat,
            lon,
            row,
            col,
            val,
        )
        return float(val)


def mosaic_and_crop(
    tif_paths: list[str],
    bounds: tuple[float, float, float, float],
    downsample_factor: int = 1,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Merge elevation tiles and crop to a bounding box.

    Args:
        tif_paths: Paths to the tiles to merge. ``.gz`` files are opened via ``vsigzip``.
        bounds: Bounding box ``(lon_min, lat_min, lon_max, lat_max)`` in WGS84.
        downsample_factor: Factor to downsample the output (default 1 = native resolution).
                           Factor N means take every Nth pixel (resolution * N).

    Returns:
        A tuple ``(array, transform)`` of the cropped DEM and its affine transform.
    """
    if not tif_paths:
        raise ValueError("No elevation tiles provided for mosaicing.")

    # Pre-process inputs:
    # If a file is large (like SRTM background) and we only need a small slice,
    # we should crop it to 'bounds' *before* merging.
    # Otherwise, merging a 1x1 degree SRTM tile at 2m (Swiss) resolution creates a 50GB array -> SIGKILL.

    from rasterio.io import MemoryFile
    from rasterio.windows import from_bounds as window_from_bounds

    # We need to manage the lifecycle of MemoryFiles so they don't close before merge
    memfiles = []

    src_files = []

    # Pad bounds slightly to avoid edge artifacts
    pad = 0.001  # ~100m
    padded_bounds = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    try:
        for p in tif_paths:
            src = (
                rasterio.open(f"/vsigzip/{p}")
                if p.endswith(".gz")
                else rasterio.open(p)
            )

            # Check if file is "large" (e.g. SRTM) and significantly larger than our bounds
            # SRTM is 1x1 deg. Slice is usually 0.05 deg.
            # Simple heuristic: if src bounds area > 10x target bounds area -> Crop it.
            src_area = (src.bounds.right - src.bounds.left) * (
                src.bounds.top - src.bounds.bottom
            )
            target_area = (padded_bounds[2] - padded_bounds[0]) * (
                padded_bounds[3] - padded_bounds[1]
            )

            if src_area > 10 * target_area:
                try:
                    # Calculate window
                    win = window_from_bounds(*padded_bounds, transform=src.transform)

                    # Clip window to actual file dimensions
                    win = win.intersection(
                        rasterio.windows.Window(0, 0, src.width, src.height)
                    )

                    if win.width > 0 and win.height > 0:
                        logger.debug(f"Pre-cropping large raster {p} to {win}")
                        data = src.read(window=win)
                        win_transform = rasterio.windows.transform(win, src.transform)

                        # Create in-memory file for the cropped chunk
                        memfile = MemoryFile()
                        dst = memfile.open(
                            driver="GTiff",
                            height=int(win.height),
                            width=int(win.width),
                            count=src.count,
                            dtype=src.dtypes[0],
                            crs=src.crs,
                            transform=win_transform,
                            nodata=src.nodata,
                        )
                        dst.write(data)
                        memfiles.append(memfile)  # Keep alive
                        # Close original source, append new source
                        src.close()
                        src_files.append(dst)
                    else:
                        logger.debug(
                            f"Raster {p} does not intersect target bounds, skipping."
                        )
                        src.close()
                except Exception as e:
                    logger.warning(f"Failed to pre-crop {p}: {e}. Using full file.")
                    src_files.append(src)
            else:
                src_files.append(src)

        if not src_files:
            raise ValueError("No elevation tiles intersect the requested bounds.")

        # Determine target resolution
        # By default, merge uses the res of the first file.
        # Since we prepend SRTM (low res) to fallback, we MUST scan for the highest resolution (smallest pixel)
        # and enforce it.

        # helper to mean of x_res and y_res
        def get_avg_res(src):
            return (abs(src.res[0]) + abs(src.res[1])) / 2

        # Find the source with the finest resolution (smallest number)
        best_src = min(src_files, key=get_avg_res)
        native_res = best_src.res
        logger.info(
            f"Mosaic selected best resolution: {native_res} from source index {src_files.index(best_src)}"
        )

        res = None
        if downsample_factor > 1:
            rx, ry = native_res
            res = (rx * downsample_factor, ry * downsample_factor)
            logger.debug(
                f"Downsampling mosaic by {downsample_factor}x. Target res: {res}"
            )
        else:
            # Force the best resolution found (in case first file is low-res SRTM)
            res = native_res

        # Debug: log bounds and find best dtype
        best_dtype = rasterio.int16
        for i, src in enumerate(src_files):
            logger.debug(
                f"Source {i} bounds: {src.bounds}, crs: {src.crs}, res: {src.res}, dtype: {src.dtypes[0]}"
            )
            if "float" in src.dtypes[0]:
                best_dtype = rasterio.float32

        logger.debug(f"Mosaic merging with dtype: {best_dtype}")

        lon_min, lat_min, lon_max, lat_max = bounds
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min
        bounds = (lon_min, lat_min, lon_max, lat_max)

        mosaic = None
        transform = None
        if not is_antimeridian_crossing(bounds):
            res_x = abs(res[0])
            res_y = abs(res[1])
            pad = max(res_x, res_y) * 2
            merge_bounds = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)
            try:
                mosaic, transform = merge(
                    src_files, bounds=merge_bounds, res=res, dtype=best_dtype
                )
            except Exception as e:
                logger.warning(
                    f"Bounded merge failed ({e}); falling back to un-bounded merge."
                )
                mosaic = None

        if mosaic is None:
            mosaic, transform = merge(src_files, res=res, dtype=best_dtype)

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

        row_off = max(0, int(window.row_off))
        row_end = min(mosaic.shape[1], int(window.row_off) + int(window.height))
        col_off = max(0, int(window.col_off))
        col_end = min(mosaic.shape[2], int(window.col_off) + int(window.width))

        clamped_win = rasterio.windows.Window(
            col_off, row_off, max(0, col_end - col_off), max(0, row_end - row_off)
        )
        clipped = mosaic[:, row_off:row_end, col_off:col_end]
        cropped_transform = round_affine(
            rasterio.windows.transform(clamped_win, transform)
        )
        return clipped[0], cropped_transform
    finally:
        for src in src_files:
            try:
                src.close()
            except Exception:
                pass
        for mf in memfiles:
            try:
                mf.close()
            except Exception:
                pass


# Elevation Downloading Utilities

TILE_CACHE_DIR = settings.TILE_CACHE_DIR
ALTI3D_CACHE_DIR = settings.ALTI3D_CACHE_DIR
ALTI3D_DOWNLOADER = settings.ALTI3D_DOWNLOADER

# Swiss bounds (lon_min, lat_min, lon_max, lat_max)
SWISS_BOUNDS = (5.95, 45.82, 10.49, 47.81)


def is_antimeridian_crossing(bounds: tuple[float, float, float, float]) -> bool:
    """Checks if a bounding box crosses the antimeridian.

    Args:
        bounds (Tuple[float, float, float, float]): Bounding box as (lon_min, lat_min, lon_max, lat_max).

    Returns:
        bool: True if the bounding box crosses the antimeridian, False otherwise.
    """
    lon_min, _, lon_max, _ = bounds
    return lon_min > lon_max


def _download_tiles(
    bounds: tuple[float, float, float, float],
) -> list[str]:
    """
    Downloads and caches SRTM tiles intersecting the given bounding box from OpenTopography AWS S3.
    Tiles are stored in gzip format as .hgt.gz files.

    Args:
        bounds (tuple): Geographic bounding box (lon_min, lat_min, lon_max, lat_max).

    Returns:
        list[str]: Paths to downloaded and cached SRTM .hgt.gz files.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    lat_range = range(floor(lat_min), floor(lat_max) + 1)
    lon_range = range(floor(lon_min), floor(lon_max) + 1)

    TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    new_tiles_downloaded = False
    for lat in lat_range:
        for lon in lon_range:
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            filename = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}.hgt.gz"
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{abs(lat):02d}/{filename}"
            local_path = TILE_CACHE_DIR / filename
            lock_path = str(local_path) + ".lock"
            with FileLock(lock_path, timeout=60):
                if not local_path.exists():
                    logger.info(f"Downloading {url} → {local_path}")
                    tmp_path = local_path.with_suffix(".tmp.gz")
                    try:
                        r = requests.get(url, stream=True, timeout=30)
                        if r.status_code == 200:
                            with open(tmp_path, "wb") as f:
                                f.writelines(r.iter_content(chunk_size=8192))
                            tmp_path.replace(local_path)
                            touch_cache_file(local_path)
                            new_tiles_downloaded = True
                        else:
                            logger.warning(
                                f"Failed to fetch {url}: HTTP {r.status_code}"
                            )
                            continue
                    except Exception:
                        tmp_path.unlink(missing_ok=True)
                        raise
                else:
                    logger.debug(f"Tile already exists: {local_path}")
                    touch_cache_file(local_path)
                paths.append(str(local_path))
    if new_tiles_downloaded:
        enforce_cache_limit(
            TILE_CACHE_DIR,
            int(getattr(settings, "TILE_CACHE_MAX_MB", 1024)) * 1024 * 1024,
        )
    return paths


def download_srtm_tiles_for_bounds(
    bounds: tuple[float, float, float, float],
) -> list[str]:
    """Downloads SRTM tiles for the given bounding box.
    Handles antimeridian crossing by splitting the bounding box.
    Args:
        bounds (tuple): Geographic bounding box (lon_min, lat_min, lon_max, lat_max).
    Returns:
        list[str]: Paths to downloaded and cached SRTM .hgt.gz files.
    """
    if is_antimeridian_crossing(bounds):  # some fancy recursive action
        logger.warning("Antimeridian crossing detected; splitting bounds.")
        lon_min, lat_min, lon_max, lat_max = bounds
        paths1 = download_srtm_tiles_for_bounds((lon_min, lat_min, 180.0, lat_max))
        paths2 = download_srtm_tiles_for_bounds((-180.0, lat_min, lon_max, lat_max))
        return paths1 + paths2
    return _download_tiles(bounds)


def get_srtm_tile_path(lat: float, lon: float, tile_cache_dir: Path) -> Path:
    """Get the local path for a specific SRTM tile based on latitude and longitude.
    Args:
        lat (float): Latitude of the tile.
        lon (float): Longitude of the tile.
        tile_cache_dir (Path): Directory where tiles are cached.
    Returns:
        Path: Local path to the SRTM tile file.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    lat_int = floor(lat)
    lon_int = floor(lon)
    filename = f"{ns}{abs(lat_int):02d}{ew}{abs(lon_int):03d}.hgt.gz"
    return tile_cache_dir / filename


def ensure_tile_downloaded(lat: float, lon: float) -> Path | None:
    """Ensure that the SRTM tile for the given latitude and longitude is downloaded.
    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
    Returns:
        Path: Local path to the downloaded SRTM tile file, or None if unavailable.
    """
    # Use a minimal bounding box that covers only the containing tile
    lat_int = floor(lat)
    lon_int = floor(lon)
    bounds = (lon_int, lat_int, lon_int + 1, lat_int + 1)

    paths = download_srtm_tiles_for_bounds(bounds)
    if not paths:
        return None
    return Path(paths[0])


def within_swiss_bounds(bounds: tuple[float, float, float, float]) -> bool:
    """Check if bounds intersect Switzerland."""
    lon_min, lat_min, lon_max, lat_max = bounds
    s_lon_min, s_lat_min, s_lon_max, s_lat_max = SWISS_BOUNDS
    return not (
        lon_max < s_lon_min
        or lon_min > s_lon_max
        or lat_max < s_lat_min
        or lat_min > s_lat_max
    )


def download_alti3d_tiles_for_bounds(
    bounds: tuple[float, float, float, float],
) -> list[str]:
    """Download SwissALTI3D tiles using native STAC API implementation."""
    return download_swissalti3d_tiles(bounds)


def download_elevation_tiles_for_bounds(
    bounds: tuple[float, float, float, float],
) -> list[str]:
    """Download elevation tiles, preferring SwissALTI3D when in Switzerland."""
    # Always fetch SRTM as a fallback/background layer to prevent holes
    srtm_tiles = download_srtm_tiles_for_bounds(bounds)

    if within_swiss_bounds(bounds):
        swiss_tiles = download_alti3d_tiles_for_bounds(bounds)
        if swiss_tiles:
            logger.info(
                f"Merging {len(swiss_tiles)} SwissALTI3D tiles with {len(srtm_tiles)} SRTM tiles."
            )
            # Return SRTM first (background), then Swiss (foreground)
            # rasterio.merge paints in order, so later files overwrite earlier ones.
            return srtm_tiles + swiss_tiles
        logger.info("No SwissALTI3D tiles found, using SRTM only.")

    return srtm_tiles


def get_elevation_stats_fast(
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """
    Get approximate min and max elevation for the given bounds using SRTM data.
    Uses downsampling for speed.
    """
    # Force SRTM only
    tiles = download_srtm_tiles_for_bounds(bounds)
    if not tiles:
        return 0.0, 0.0

    # Downsample significantly for speed (e.g. 10x = 1/100th of pixels)
    mosaic, _ = mosaic_and_crop(tiles, bounds, downsample_factor=10)

    # Filter invalid values
    valid_mask = (mosaic > -10000) & (mosaic < 10000)
    if not valid_mask.any():
        return 0.0, 0.0

    valid_data = mosaic[valid_mask]
    return float(valid_data.min()), float(valid_data.max())
