import logging

from django.conf import settings
from django.utils.text import slugify

logger = logging.getLogger(__name__)


def build_export_basename(
    address: str, coords: list[float], height_mm: str, num_layers: int
) -> str:
    """Build a consistent export filename from location, height, and layer count.

    Args:
        address (str): User-specified address or location name.
        coords (list[float]): Coordinates [longitude, latitude] if no address.
        height_mm (str): Height of each layer in millimeters as a string.
        num_layers (int): Total number of layers.

    Returns:
        str: A slug-safe export filename.
    """
    if address:
        location_slug = slugify(address)
    elif coords and isinstance(coords, list) and len(coords) == 2:
        location_slug = f"{coords[0]:.4f}_{coords[1]:.4f}"
    else:
        location_slug = "contours"
    logger.debug(f"zip filename: {location_slug}_{height_mm}mm_{num_layers}layers")
    return f"{location_slug}_{height_mm}mm_{num_layers}layers"
