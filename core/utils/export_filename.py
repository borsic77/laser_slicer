import logging
import re
import unicodedata
from venv import logger

from django.conf import settings

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "_", value)


def build_export_basename(
    address: str, coords: list[float], height_mm: str, num_layers: int
) -> str:
    if address:
        location_slug = slugify(address)
    elif coords and isinstance(coords, list) and len(coords) == 2:
        location_slug = f"{coords[0]:.4f}_{coords[1]:.4f}"
    else:
        location_slug = "contours"
    logger.debug(f"zip filename: {location_slug}_{height_mm}mm_{num_layers}layers")
    return f"{location_slug}_{height_mm}mm_{num_layers}layers"
