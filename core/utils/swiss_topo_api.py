import io
import logging
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import requests
from django.conf import settings
from filelock import FileLock
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, calculate_default_transform, reproject

try:
    from core.utils.cache_ops import enforce_cache_limit, touch_cache_file
except ImportError:  # pragma: no cover

    def touch_cache_file(path):
        pass

    def enforce_cache_limit(cache_dir, max_bytes):
        pass


logger = logging.getLogger(__name__)

STAC_API_URL = (
    "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
)
# We prefer 2m resolution for reasonable file sizes and processing time
# 0.5m files are significantly larger.
PREFERRED_RESOLUTION = "2"

ALTI3D_CACHE_DIR = settings.ALTI3D_CACHE_DIR

# 22 lakes from swisstopo collection ch.swisstopo.swissbathy3d
# BBoxes in WGS84 (lon_min, lat_min, lon_max, lat_max)
# Surface elevations in meters above sea level (LHN95/LN02 datum)
SWISS_BATHY3D_LAKES = [
    {
        "id": "swissbathy3d_aegerisee",
        "name": "Aegerisee",
        "bbox": (8.5846, 47.0976, 8.6516, 47.1433),
        "surface_elevation": 724.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_aegerisee/swissbathy3d_aegerisee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_baldeggersee",
        "name": "Baldeggersee",
        "bbox": (8.2432, 47.1774, 8.2770, 47.2174),
        "surface_elevation": 463.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_baldeggersee/swissbathy3d_baldeggersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_bielersee",
        "name": "Lake Biel (Bielersee)",
        "bbox": (7.0695, 47.0314, 7.2413, 47.1399),
        "surface_elevation": 429.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_bielersee/swissbathy3d_bielersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_bodensee",
        "name": "Lake Constance (Bodensee)",
        "bbox": (8.8450, 47.4674, 9.7752, 47.8240),
        "surface_elevation": 395.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_bodensee/swissbathy3d_bodensee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_brienzersee",
        "name": "Lake Brienz (Brienzersee)",
        "bbox": (7.8568, 46.6795, 8.0539, 46.7615),
        "surface_elevation": 564.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_brienzersee/swissbathy3d_brienzersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_hallwilersee",
        "name": "Hallwilersee",
        "bbox": (8.1803, 47.2321, 8.2332, 47.3216),
        "surface_elevation": 449.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_hallwilersee/swissbathy3d_hallwilersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_lacdejoux",
        "name": "Lac de Joux",
        "bbox": (6.2493, 46.6120, 6.3421, 46.6760),
        "surface_elevation": 1004.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_lacdejoux/swissbathy3d_lacdejoux_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_lacleman",
        "name": "Lake Geneva (Lac Léman)",
        "bbox": (6.1223, 46.1968, 6.9334, 46.5272),
        "surface_elevation": 372.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_lacleman/swissbathy3d_lacleman_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_lacneuchatel",
        "name": "Lake Neuchâtel",
        "bbox": (6.6227, 46.7772, 7.0720, 47.0495),
        "surface_elevation": 429.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_lacneuchatel/swissbathy3d_lacneuchatel_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_lagomaggiore",
        "name": "Lake Maggiore",
        "bbox": (8.3146, 45.6798, 8.9107, 46.2594),
        "surface_elevation": 193.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_lagomaggiore/swissbathy3d_lagomaggiore_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_lungernsee",
        "name": "Lungerersee",
        "bbox": (8.1499, 46.7827, 8.1757, 46.8135),
        "surface_elevation": 688.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_lungernsee/swissbathy3d_lungernsee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_murtensee",
        "name": "Lake Murten (Murtensee)",
        "bbox": (7.0313, 46.9053, 7.1368, 46.9597),
        "surface_elevation": 429.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_murtensee/swissbathy3d_murtensee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_rotsee",
        "name": "Rotsee",
        "bbox": (8.2999, 47.0630, 8.3285, 47.0763),
        "surface_elevation": 419.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_rotsee/swissbathy3d_rotsee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_sarnersee",
        "name": "Sarnersee",
        "bbox": (8.1888, 46.8454, 8.2440, 46.8900),
        "surface_elevation": 469.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_sarnersee/swissbathy3d_sarnersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_sempachersee",
        "name": "Sempachersee",
        "bbox": (8.1213, 47.1178, 8.1922, 47.1739),
        "surface_elevation": 504.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_sempachersee/swissbathy3d_sempachersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_silsersee",
        "name": "Lake Sils (Silsersee)",
        "bbox": (9.6953, 46.4025, 9.7657, 46.4416),
        "surface_elevation": 1797.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_silsersee/swissbathy3d_silsersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_silvaplanersee",
        "name": "Lake Silvaplana (Silvaplanersee)",
        "bbox": (9.7685, 46.4329, 9.8078, 46.4605),
        "surface_elevation": 1791.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_silvaplanersee/swissbathy3d_silvaplanersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_thunersee",
        "name": "Lake Thun (Thunersee)",
        "bbox": (7.6215, 46.6535, 7.8444, 46.7531),
        "surface_elevation": 558.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_thunersee/swissbathy3d_thunersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_vierwaldstaettersee",
        "name": "Lake Lucerne (Vierwaldstättersee)",
        "bbox": (8.2787, 46.8884, 8.6251, 47.0822),
        "surface_elevation": 434.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_vierwaldstaettersee/swissbathy3d_vierwaldstaettersee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_walensee",
        "name": "Walensee",
        "bbox": (9.0953, 47.0798, 9.3218, 47.1575),
        "surface_elevation": 419.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_walensee/swissbathy3d_walensee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_zuerichsee",
        "name": "Lake Zurich (Zürichsee)",
        "bbox": (8.5207, 47.1946, 8.8288, 47.3688),
        "surface_elevation": 406.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_zuerichsee/swissbathy3d_zuerichsee_2056_5728.esriasciigrid.zip",
    },
    {
        "id": "swissbathy3d_zugersee",
        "name": "Lake Zug (Zugersee)",
        "bbox": (8.4389, 47.0538, 8.5339, 47.1896),
        "surface_elevation": 413.0,
        "zip_url": "https://data.geo.admin.ch/ch.swisstopo.swissbathy3d/swissbathy3d_zugersee/swissbathy3d_zugersee_2056_5728.esriasciigrid.zip",
    },
]


def get_swiss_bathy3d_lakes_for_bounds(
    bounds: tuple[float, float, float, float],
) -> list[dict]:
    """Return all SwissBATHY3D lakes intersecting the given WGS84 bounding box."""
    lon_min, lat_min, lon_max, lat_max = bounds
    matching = []
    for lake in SWISS_BATHY3D_LAKES:
        l_min_lon, l_min_lat, l_max_lon, l_max_lat = lake["bbox"]
        if not (
            lon_max < l_min_lon
            or lon_min > l_max_lon
            or lat_max < l_min_lat
            or lat_min > l_max_lat
        ):
            matching.append(lake)
    return matching


def get_swiss_lake_surface_elevation(
    bounds: tuple[float, float, float, float],
) -> float | None:
    """Return surface elevation in meters for the primary Swiss lake intersecting bounds, or None."""
    lakes = get_swiss_bathy3d_lakes_for_bounds(bounds)
    if not lakes:
        return None
    if len(lakes) == 1:
        return float(lakes[0]["surface_elevation"])

    # If multiple lakes intersect, pick the one with the largest intersection area with bounds
    lon_min, lat_min, lon_max, lat_max = bounds
    best_lake = None
    best_area = -1.0
    for lake in lakes:
        l_min_lon, l_min_lat, l_max_lon, l_max_lat = lake["bbox"]
        i_min_lon = max(lon_min, l_min_lon)
        i_max_lon = min(lon_max, l_max_lon)
        i_min_lat = max(lat_min, l_min_lat)
        i_max_lat = min(lat_max, l_max_lat)
        area = max(0.0, i_max_lon - i_min_lon) * max(0.0, i_max_lat - i_min_lat)
        if area > best_area:
            best_area = area
            best_lake = lake

    return float(best_lake["surface_elevation"]) if best_lake else None


class HttpRangeFile(io.RawIOBase):
    """File-like object that supports random-access reads over HTTP Range requests."""

    def __init__(self, url: str, timeout: int = 20):
        self.url = url
        self.timeout = timeout
        req = urllib.request.Request(
            url, headers={"User-Agent": "laser_slicer/3.1"}, method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            self.size = int(r.headers.get("Content-Length", 0))
        self.pos = 0
        # Pre-cache tail (last 512KB) which contains the ZIP central directory
        self._tail_offset = max(0, self.size - 512 * 1024)
        req_tail = urllib.request.Request(
            url,
            headers={
                "User-Agent": "laser_slicer/3.1",
                "Range": f"bytes={self._tail_offset}-{self.size - 1}",
            },
        )
        with urllib.request.urlopen(req_tail, timeout=timeout) as r:
            self._tail = r.read()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        length = min(len(b), self.size - self.pos)
        if self.pos >= self._tail_offset and self._tail is not None:
            rel = self.pos - self._tail_offset
            chunk = self._tail[rel : rel + length]
        else:
            req = urllib.request.Request(
                self.url,
                headers={
                    "User-Agent": "laser_slicer/3.1",
                    "Range": f"bytes={self.pos}-{self.pos + length - 1}",
                },
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                chunk = r.read()
        b[: len(chunk)] = chunk
        self.pos += len(chunk)
        return len(chunk)


def download_swissbathy3d_tiles(
    bounds: tuple[float, float, float, float],
) -> list[str]:
    """Downloads and reprojects SwissBATHY3D lake bathymetry tiles intersecting bounds.

    Args:
        bounds: (lon_min, lat_min, lon_max, lat_max) in WGS84.

    Returns:
        List of local file paths to reprojected WGS84 GeoTIFF files.
    """
    lakes = get_swiss_bathy3d_lakes_for_bounds(bounds)
    if not lakes:
        return []

    ALTI3D_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bounds
    try:
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:2056", always_xy=True
        )
        e1, n1 = transformer.transform(lon_min, lat_min)
        e2, n2 = transformer.transform(lon_max, lat_max)
        e_min, e_max = min(e1, e2), max(e1, e2)
        n_min, n_max = min(n1, n2), max(n1, n2)
        e_km_range = range(
            int(math.floor(e_min / 1000.0)), int(math.floor(e_max / 1000.0)) + 1
        )
        n_km_range = range(
            int(math.floor(n_min / 1000.0)), int(math.floor(n_max / 1000.0)) + 1
        )
        needed_tile_coords = {(e, n) for e in e_km_range for n in n_km_range}
    except Exception as e:
        logger.error(f"Failed to transform bounds to EPSG:2056: {e}")
        return []

    downloaded_paths = []
    new_tiles_created = False
    src_crs = CRS.from_epsg(2056)
    dst_crs = CRS.from_epsg(4326)

    for lake in lakes:
        lake_id = lake["id"]
        # Check if all needed tiles for this lake already exist in cache
        uncached_coords = set()
        for e, n in needed_tile_coords:
            wgs84_path = ALTI3D_CACHE_DIR / f"{lake_id}_{e}_{n}_wgs84.tif"
            if wgs84_path.exists():
                touch_cache_file(wgs84_path)
                downloaded_paths.append(str(wgs84_path))
            else:
                uncached_coords.add((e, n))

        if not uncached_coords:
            continue

        try:
            logger.info(
                f"Opening SwissBATHY3D remote archive for {lake['name']} ({lake_id})..."
            )
            range_file = HttpRangeFile(lake["zip_url"], timeout=30)
            buffered = io.BufferedReader(range_file)
            with zipfile.ZipFile(buffered) as zf:
                name_list = zf.namelist()
                for e, n in uncached_coords:
                    needle = f"_{e}_{n}.asc"
                    matches = [name for name in name_list if name.endswith(needle)]
                    if not matches:
                        continue

                    asc_name = matches[0]
                    wgs84_path = ALTI3D_CACHE_DIR / f"{lake_id}_{e}_{n}_wgs84.tif"
                    lock_path = str(wgs84_path) + ".lock"

                    with FileLock(lock_path, timeout=60):
                        if wgs84_path.exists():
                            touch_cache_file(wgs84_path)
                            downloaded_paths.append(str(wgs84_path))
                            continue

                        logger.info(
                            f"Extracting and reprojecting SwissBATHY3D tile {asc_name} -> {wgs84_path.name}"
                        )
                        asc_bytes = zf.read(asc_name)
                        with MemoryFile(asc_bytes, ext=".asc") as mem:
                            with mem.open() as src:
                                raw = src.read(1)
                                # 2m target resolution (matches SwissALTI3D 2m tiles)
                                w2 = max(1, src.width // 2)
                                h2 = max(1, src.height // 2)
                                transform, width, height = calculate_default_transform(
                                    src_crs,
                                    dst_crs,
                                    w2,
                                    h2,
                                    *src.bounds,
                                )
                                dst_data = np.full(
                                    (height, width), -32768.0, dtype=np.float32
                                )
                                reproject(
                                    source=raw,
                                    destination=dst_data,
                                    src_transform=src.transform,
                                    src_crs=src_crs,
                                    src_nodata=src.nodata,
                                    dst_transform=transform,
                                    dst_crs=dst_crs,
                                    dst_nodata=-32768.0,
                                    resampling=Resampling.bilinear,
                                )

                        # Skip tiles with zero valid depth points
                        if not (dst_data > -1000).any():
                            continue

                        tmp_wgs84 = wgs84_path.with_suffix(".tmp.tif")
                        try:
                            with rasterio.open(
                                tmp_wgs84,
                                "w",
                                driver="GTiff",
                                height=height,
                                width=width,
                                count=1,
                                dtype=np.float32,
                                crs=dst_crs,
                                transform=transform,
                                nodata=-32768.0,
                                compress="deflate",
                            ) as dst:
                                dst.write(dst_data, 1)
                            tmp_wgs84.replace(wgs84_path)
                            new_tiles_created = True
                            downloaded_paths.append(str(wgs84_path))
                        except Exception as e:
                            tmp_wgs84.unlink(missing_ok=True)
                            logger.error(
                                f"Failed to write reprojected tile {wgs84_path}: {e}"
                            )
        except Exception as e:
            logger.warning(
                f"Failed to fetch SwissBATHY3D data for lake {lake['name']}: {e}"
            )

    if new_tiles_created:
        enforce_cache_limit(
            ALTI3D_CACHE_DIR,
            int(getattr(settings, "ALTI3D_CACHE_MAX_MB", 1024)) * 1024 * 1024,
        )

    return downloaded_paths


def download_swissalti3d_tiles(bounds: tuple[float, float, float, float]) -> list[str]:
    """
    Downloads SwissALTI3D tiles covering the given bounding box using the STAC API.

    Args:
        bounds: (lon_min, lat_min, lon_max, lat_max) in WGS84.

    Returns:
        List of local file paths to the downloaded TIFF files.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    ALTI3D_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Query STAC API
    params = {
        "bbox": bbox_str,
        # Increase limit to avoid pagination for reasonable areas (max usually 50-100)
        "limit": 100,
    }

    downloaded_paths = []

    try:
        logger.info(f"Querying SwissTopo STAC API with bbox={bbox_str}")
        resp = requests.get(STAC_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("features", [])
        if not items:
            logger.warning("No SwissALTI3D items found for these bounds.")
            return []

        logger.info(f"Found {len(items)} matching tiles.")

        for item in items:
            # Find the correct asset
            asset_url = None
            # Filter assets for 2m resolution TIF
            # Looking for keys like "swissalti3d_..._2_... .tif" or checking eo:gsd
            for key, asset in item["assets"].items():
                # Check for TIF mimetype
                if "image/tiff" not in asset["type"]:
                    continue

                # Robust check for resolution:
                # 1. Check eo:gsd property if available
                # 2. Check filename pattern
                gsd = asset.get("eo:gsd")
                href = asset["href"]

                # We want 2.0 resolution
                if gsd == 2.0:
                    asset_url = href
                    break
                elif f"_{PREFERRED_RESOLUTION}_" in href and href.endswith(".tif"):
                    # Fallback pattern matching
                    asset_url = href
                    break

            if not asset_url:
                logger.warning(
                    f"No suitable {PREFERRED_RESOLUTION}m asset found for item {item['id']}"
                )
                continue

            # Determine local filename
            filename = asset_url.split("/")[-1]
            local_path = ALTI3D_CACHE_DIR / filename
            wgs84_path = local_path.with_name(local_path.stem + "_wgs84.tif")
            lock_path = str(local_path) + ".lock"

            with FileLock(lock_path, timeout=120):
                if not local_path.exists():
                    logger.info(f"Downloading {asset_url} -> {local_path}")
                    tmp_local = local_path.with_suffix(".tmp.tif")
                    try:
                        with requests.get(asset_url, stream=True, timeout=60) as r:
                            r.raise_for_status()
                            with open(tmp_local, "wb") as f:
                                f.writelines(r.iter_content(chunk_size=8192))
                        tmp_local.replace(local_path)
                    except Exception:
                        tmp_local.unlink(missing_ok=True)
                        raise
                else:
                    logger.debug(f"Tile already cached: {local_path}")
                    touch_cache_file(local_path)

                if not wgs84_path.exists():
                    logger.info(f"Reprojecting {local_path} to WGS84 -> {wgs84_path}")
                    tmp_wgs84 = wgs84_path.with_suffix(".tmp.tif")
                    try:
                        reproject_to_wgs84(local_path, tmp_wgs84)
                        tmp_wgs84.replace(wgs84_path)
                    except Exception as e:
                        tmp_wgs84.unlink(missing_ok=True)
                        logger.error(f"Failed to reproject {local_path}: {e}")
                        continue
                else:
                    touch_cache_file(wgs84_path)

            downloaded_paths.append(str(wgs84_path))

    except requests.RequestException as e:
        logger.error(f"Failed to query or download from SwissTopo API: {e}")
        raise RuntimeError(f"SwissTopo API error: {e}")

    enforce_cache_limit(
        ALTI3D_CACHE_DIR,
        int(getattr(settings, "ALTI3D_CACHE_MAX_MB", 1024)) * 1024 * 1024,
    )

    return downloaded_paths


def reproject_to_wgs84(src_path: Path, dst_path: Path):
    """Reproject a raster to EPSG:4326 (WGS84)."""
    dst_crs = "EPSG:4326"

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        nodata = src.nodata if src.nodata is not None else -32768.0
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": nodata,
            }
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    dst_nodata=nodata,
                    resampling=Resampling.bilinear,
                )
