import logging

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
    location_slug = slugify(address) if address else ""
    if not location_slug:
        if coords and isinstance(coords, (list, tuple)) and len(coords) == 2:
            try:
                location_slug = f"{float(coords[0]):.4f}_{float(coords[1]):.4f}"
            except (TypeError, ValueError):
                location_slug = "contours"
        else:
            location_slug = "contours"

    try:
        height_val = float(height_mm)
        safe_height = f"{height_val:g}"
    except (TypeError, ValueError):
        safe_height = slugify(str(height_mm)) or "unknown"

    safe_layers = int(num_layers)
    result = f"{location_slug}_{safe_height}mm_{safe_layers}layers"
    logger.debug("zip filename: %s", result)
    return result
