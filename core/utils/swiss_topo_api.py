import logging
from pathlib import Path

import rasterio
import requests
from django.conf import settings
from filelock import FileLock
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

        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
