import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import requests
from django.conf import settings
from django.core.cache import cache
from pyproj import Transformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def get_transformer_for_coords(center_x: float, center_y: float) -> Transformer:
    """Get a pyproj Transformer for converting WGS84 to UTM coordinates.
    This function caches the transformer for different center coordinates to improve performance.
    Args:
        center_x (float): Longitude of the center point.
        center_y (float): Latitude of the center point.
    Returns:
        Transformer: A pyproj Transformer object for the specified coordinates.
    """
    zone_number = int((center_x + 180) / 6) + 1
    is_northern = center_y >= 0
    epsg_code = f"326{zone_number:02d}" if is_northern else f"327{zone_number:02d}"
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)


@dataclass
class Coordinates:
    lat: float
    lon: float


def geocode_address(address: str) -> Coordinates:
    """Resolve a textual address to geographic coordinates using Nominatim.

    This function throttles requests to respect a minimum interval between queries.

    Args:
        address (str): The address to geocode.

    Returns:
        Coordinates: A dataclass containing latitude and longitude.

    Raises:
        ValueError: If no matching result is found.
        requests.RequestException: For network or HTTP errors.
    """
    key = "last_geocode_time"
    last = cache.get(key)
    now = time.time()
    if last and now - last < 1.0:
        time.sleep(1.0 - (now - last))
    cache.set(key, now, timeout=10)
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": settings.NOMINATIM_USER_AGENT}
    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"Address not found: {address}")
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    logger.info(f"Geocoded {address} to ({lat}, {lon})")
    return Coordinates(lat=lat, lon=lon)


def compute_utm_bounds_from_wgs84(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    center_x: float,
    center_y: float,
) -> tuple[float, float, float, float]:
    """Convert WGS84 bounding box to projected UTM bounds.

    Args:
        lon_min (float): Minimum longitude of bounding box.
        lat_min (float): Minimum latitude of bounding box.
        lon_max (float): Maximum longitude of bounding box.
        lat_max (float): Maximum latitude of bounding box.
        center_x (float): Center longitude used to determine UTM zone.
        center_y (float): Center latitude used to determine UTM zone.

    Returns:
        tuple[float, float, float, float]: Bounding box in UTM coordinates as (min_x, min_y, max_x, max_y).
    """
    transformer = get_transformer_for_coords(center_x, center_y)
    utm_minx, utm_miny = transformer.transform(lon_min, lat_min)
    utm_maxx, utm_maxy = transformer.transform(lon_max, lat_max)
    return (utm_minx, utm_miny, utm_maxx, utm_maxy)


DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
DEBUG = settings.DEBUG
