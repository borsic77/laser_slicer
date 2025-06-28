import logging
import os
import subprocess
from math import floor
from pathlib import Path
from typing import List, Tuple

import requests
from django.conf import settings
from filelock import FileLock

logger = logging.getLogger(__name__)
TILE_CACHE_DIR = settings.TILE_CACHE_DIR
ALTI3D_CACHE_DIR = settings.ALTI3D_CACHE_DIR
ALTI3D_DOWNLOADER = settings.ALTI3D_DOWNLOADER

# Swiss bounds (lon_min, lat_min, lon_max, lat_max)
SWISS_BOUNDS = (5.95, 45.82, 10.49, 47.81)


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


def ensure_tile_downloaded(lat: float, lon: float) -> Path:
    """Ensure that the SRTM tile for the given latitude and longitude is downloaded.
    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
    Returns:
        Path: Local path to the downloaded SRTM tile file.
    """
    # Use a minimal bounding box that covers only the containing tile
    lat_int = floor(lat)
    lon_int = floor(lon)
    bounds = (lon_int, lat_int, lon_int + 1, lat_int + 1)

    paths = download_srtm_tiles_for_bounds(bounds)
    # There will be exactly one path in the returned list, matching our point
    return Path(paths[0])


def within_swiss_bounds(bounds: Tuple[float, float, float, float]) -> bool:
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
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    """Download SwissALTI3D tiles using the ``alti3d-downloader`` CLI."""
    lon_min, lat_min, lon_max, lat_max = bounds
    ALTI3D_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        ALTI3D_DOWNLOADER,
        "--bbox",
        f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "--resolution",
        "2",
        "--out",
        str(ALTI3D_CACHE_DIR),
    ]
    logger.info("Running %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("alti3d-downloader failed: %s", exc)
    return [str(p) for p in ALTI3D_CACHE_DIR.glob("*.tif")]


def download_elevation_tiles_for_bounds(
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    """Download elevation tiles, preferring SwissALTI3D when in Switzerland."""
    if within_swiss_bounds(bounds):
        return download_alti3d_tiles_for_bounds(bounds)
    return download_srtm_tiles_for_bounds(bounds)
