import concurrent.futures
import io
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from rasterio.transform import from_bounds

try:
    from django.conf import settings
except ImportError:  # pragma: no cover
    settings = None

try:
    from core.utils.cache_ops import enforce_cache_limit, touch_cache_file
except ImportError:  # pragma: no cover

    def touch_cache_file(path):
        pass

    def enforce_cache_limit(cache_dir, max_bytes):
        pass


logger = logging.getLogger(__name__)

# Constants for Web Mercator
EARTH_RADIUS = 6378137.0
ORIGIN_SHIFT = 2 * math.pi * EARTH_RADIUS / 2.0
INITIAL_RESOLUTION = 2 * math.pi * EARTH_RADIUS / 256.0


def resolution_at_zoom(zoom):
    return INITIAL_RESOLUTION / (2**zoom)


def tile_to_web_mercator(xtile, ytile, zoom):
    """Returns (min_x, min_y, max_x, max_y) in EPSG:3857 for a given tile."""
    res = resolution_at_zoom(zoom)
    min_x = xtile * 256 * res - ORIGIN_SHIFT
    max_y = ORIGIN_SHIFT - ytile * 256 * res
    max_x = (xtile + 1) * 256 * res - ORIGIN_SHIFT
    min_y = ORIGIN_SHIFT - (ytile + 1) * 256 * res
    return min_x, min_y, max_x, max_y


def lat_lon_to_tile(lat, lon, zoom):
    n = int(2**zoom)
    lat_c = max(-85.05112878, min(85.05112878, float(lat)))
    lon_c = max(-180.0, min(180.0, float(lon)))
    xtile = int((lon_c + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(math.radians(lat_c))) / math.pi) / 2.0 * n)
    return (max(0, min(n - 1, xtile)), max(0, min(n - 1, ytile)))


# Tile Providers
TILE_PROVIDERS = {
    "osm": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "water_color": (170, 211, 223),  # #AAD3DF
        "attribution": "© OpenStreetMap contributors",
    },
    "cartodb_voyager_nolabels": {
        "url": "https://a.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
        "water_color": (213, 232, 235),  # #D5E8EB
        "attribution": "© CartoDB, © OpenStreetMap contributors",
    },
}


def fetch_osm_static_image(bounds, zoom=15, provider="osm"):
    """
    Fetch and stitch map tiles for a given bounding box.
    Returns (PIL.Image, rasterio.Affine, str_crs).

    Args:
        bounds: (lon_min, lat_min, lon_max, lat_max)
        zoom: Zoom level
        provider: Key in TILE_PROVIDERS or custom URL template.
    """
    lon_min, lat_min, lon_max, lat_max = bounds

    # Resolve provider URL
    if provider in TILE_PROVIDERS:
        url_template = TILE_PROVIDERS[provider]["url"]
    else:
        url_template = provider

    # Adaptively step down zoom
    while zoom > 10:
        x_min, y_max = lat_lon_to_tile(lat_min, lon_min, zoom)
        x_max, y_min = lat_lon_to_tile(lat_max, lon_max, zoom)
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        if (x_max - x_min + 1) * (y_max - y_min + 1) > 36:
            zoom -= 1
        else:
            break

    # x is lon, y is lat
    x_min, y_max = lat_lon_to_tile(
        lat_min, lon_min, zoom
    )  # lat_min is bottom, so y_max_tile (higher index)
    x_max, y_min = lat_lon_to_tile(
        lat_max, lon_max, zoom
    )  # lat_max is top, so y_min_tile (lower index)

    # Swap if needed (y grows downwards in tile coords)
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    width_tiles = x_max - x_min + 1
    height_tiles = y_max - y_min + 1

    tile_size = 256
    pixel_width = width_tiles * tile_size
    pixel_height = height_tiles * tile_size

    result_image = Image.new("RGB", (pixel_width, pixel_height))

    provider_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(provider)).strip("_") or "tile"
    try:
        cache_root = Path(getattr(settings, "OSM_CACHE_DIR", Path("data/osm_cache")))
        max_bytes = int(getattr(settings, "OSM_CACHE_MAX_MB", 512)) * 1024 * 1024
    except Exception:
        cache_root = Path("data/osm_cache")
        max_bytes = 512 * 1024 * 1024

    tiles_dir = cache_root / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tiles_to_fetch = []
    new_tiles_downloaded = False

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_path = tiles_dir / f"{provider_key}_{zoom}_{x}_{y}.png"
            if tile_path.exists():
                try:
                    touch_cache_file(tile_path)
                    with Image.open(tile_path) as tile_img:
                        px = (x - x_min) * tile_size
                        py = (y - y_min) * tile_size
                        result_image.paste(tile_img, (px, py))
                    continue
                except Exception as e:
                    logger.warning(f"Failed to read cached tile {tile_path}: {e}")
            tiles_to_fetch.append((x, y))

    if tiles_to_fetch:
        headers = {"User-Agent": "LaserSlicer/1.0"}
        session = requests.Session()

        def _fetch_tile(coords):
            x_coord, y_coord = coords
            url = url_template.format(z=zoom, x=x_coord, y=y_coord)
            tile_path = tiles_dir / f"{provider_key}_{zoom}_{x_coord}_{y_coord}.png"
            tmp_path = tile_path.with_suffix(".tmp.png")
            try:
                resp = session.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    with open(tmp_path, "wb") as f:
                        f.write(resp.content)
                    os.replace(tmp_path, tile_path)
                    touch_cache_file(tile_path)
                    img = Image.open(io.BytesIO(resp.content))
                    return (x_coord, y_coord), img, True
                else:
                    logger.warning(
                        f"Failed to fetch tile {x_coord},{y_coord}: HTTP {resp.status_code}"
                    )
            except Exception as e:
                print(f"Failed to fetch tile {x_coord},{y_coord}: {e}")
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
            return (x_coord, y_coord), None, False

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for (x_coord, y_coord), tile_img, downloaded in executor.map(
                _fetch_tile, tiles_to_fetch
            ):
                if downloaded and tile_img is not None:
                    new_tiles_downloaded = True
                    px = (x_coord - x_min) * tile_size
                    py = (y_coord - y_min) * tile_size
                    result_image.paste(tile_img, (px, py))

        session.close()

    if new_tiles_downloaded:
        enforce_cache_limit(cache_root, max_bytes)

    # Calculate EPSG:3857 bounds of the STITCHED image
    # Top-Left corresponds to x_min, y_min
    min_x_tl, min_y_tl, max_x_tl, max_y_tl = tile_to_web_mercator(x_min, y_min, zoom)
    # Bottom-Right corresponds to x_max, y_max
    min_x_br, min_y_br, max_x_br, max_y_br = tile_to_web_mercator(x_max, y_max, zoom)

    # Image bounds: min_x from TL, max_y from TL, max_x from BR, min_y from BR
    img_min_x = min_x_tl
    img_max_y = max_y_tl
    img_max_x = max_x_br
    img_min_y = min_y_br

    transform = from_bounds(
        img_min_x, img_min_y, img_max_x, img_max_y, pixel_width, pixel_height
    )

    return result_image, transform, "EPSG:3857"


def generate_water_mask_from_tiles(
    image: Image.Image, water_color: tuple = (170, 211, 223)
) -> np.ndarray:
    """
    Generate a boolean mask from an OSM tile image where True = Water.

    Args:
        image: PIL Image (RGB)
        water_color: RGB tuple of the water color (default: OSM standard blue #AAD3DF -> (170, 211, 223))

    Returns:
        np.ndarray: Boolean mask (height, width), True where pixel matches water_color.
    """
    # Convert to numpy
    arr = np.array(image.convert("RGB"), dtype=np.int16)

    # Calculate difference
    # Tolerance is low because OSM tiles are usually flat colors, but compression might introduce noise?
    # PNG tiles should be exact.

    diff = np.abs(arr - np.asarray(water_color, dtype=np.int16))
    mask = np.all(diff < 10, axis=2)  # Tolerance of 10 per channel

    # Clean up the mask: Close small holes (text, lines)
    # Structure defines connectivity. Default is 3x3 cross.
    # We want to close features like text labels which might be 5-10 pixels wide.
    # Iterations=2 or 3 should be enough for typical map text.
    import scipy.ndimage

    mask = scipy.ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)

    return mask
