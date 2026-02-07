import io
import math

import numpy as np
import requests
from PIL import Image
from rasterio.transform import from_bounds

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
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return (xtile, ytile)


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

    headers = {"User-Agent": "LaserSlicer/1.0"}

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            url = url_template.format(z=zoom, x=x, y=y)
            try:
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    tile_img = Image.open(io.BytesIO(resp.content))
                    # Paste position
                    px = (x - x_min) * tile_size
                    py = (y - y_min) * tile_size
                    result_image.paste(tile_img, (px, py))
            except Exception as e:
                print(f"Failed to fetch tile {x},{y}: {e}")

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
    arr = np.array(image.convert("RGB"))

    # Calculate difference
    # Tolerance is low because OSM tiles are usually flat colors, but compression might introduce noise?
    # PNG tiles should be exact.

    diff = np.abs(arr - water_color)
    mask = np.all(diff < 10, axis=2)  # Tolerance of 10 per channel

    # Clean up the mask: Close small holes (text, lines)
    # Structure defines connectivity. Default is 3x3 cross.
    # We want to close features like text labels which might be 5-10 pixels wide.
    # Iterations=2 or 3 should be enough for typical map text.
    import scipy.ndimage

    mask = scipy.ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)

    return mask
