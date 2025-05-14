import logging
from typing import Tuple

import requests

logger = logging.getLogger(__name__)


def geocode_address(address: str) -> Tuple[float, float]:
    """
    Geocode an address to (latitude, longitude) using Nominatim.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": "laser-slicer/1.0 (boris@example.com)"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError(f"Address not found: {address}")
    lat = float(results[0]["lat"])
    lon = float(results[0]["lon"])
    logger.info(f"Geocoded {address} to ({lat}, {lon})")
    return lat, lon
