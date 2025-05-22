import logging
import os
from math import floor
from pathlib import Path
from typing import List, Tuple

import requests
from django.conf import settings
from filelock import FileLock

logger = logging.getLogger(__name__)
TILE_CACHE_DIR = settings.TILE_CACHE_DIR


# Utility function to check for antimeridian crossing
def is_antimeridian_crossing(bounds: Tuple[float, float, float, float]) -> bool:
    """Checks if a bounding box crosses the antimeridian.

    Args:
        bounds (Tuple[float, float, float, float]): Bounding box as (lon_min, lat_min, lon_max, lat_max).

    Returns:
        bool: True if the bounding box crosses the antimeridian, False otherwise.
    """
    lon_min, _, lon_max, _ = bounds
    return lon_min > lon_max


def _download_tiles(
    bounds: Tuple[float, float, float, float],
) -> List[str]:
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

    paths = []
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
                    TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    r = requests.get(url, stream=True)
                    if r.status_code == 200:
                        with open(local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP {r.status_code}")
                        continue
                else:
                    logger.debug(f"Tile already exists: {local_path}")
                paths.append(str(local_path))
    return paths


def download_srtm_tiles_for_bounds(
    bounds: Tuple[float, float, float, float],
) -> List[str]:
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
