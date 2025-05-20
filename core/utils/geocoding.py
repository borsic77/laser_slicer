import logging
import time
from dataclasses import dataclass
from typing import Tuple

import requests
from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class Coordinates:
    lat: float
    lon: float


def geocode_address(address: str) -> Coordinates:
    """
    Geocode an address to (latitude, longitude) using Nominatim.
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
