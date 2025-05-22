import logging

import numpy as np

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.slicer import mosaic_and_crop

logger = logging.getLogger(__name__)


class ElevationDataError(ValueError):
    """Raised when elevation data is missing or invalid within a bounding box."""

    pass


class ElevationRangeJob:
    def __init__(self, bounds: tuple[float, float, float, float]):
        """
        Initializes the elevation range job.
        Args:
            bounds: A tuple containing the bounding box coordinates (lon_min, lat_min, lon_max, lat_max).
        """

        self.bounds = bounds

    def run(self) -> dict:
        """
        Computes the minimum and maximum elevation within the specified bounding box.
        Returns:
            A dictionary with 'min' and 'max' elevation values.
        """
        lon_min, lat_min, lon_max, lat_max = self.bounds
        tile_paths = download_srtm_tiles_for_bounds(
            (lon_min, lat_min, lon_max, lat_max)
        )
        elevation, _ = mosaic_and_crop(tile_paths, (lon_min, lat_min, lon_max, lat_max))

        if elevation.size == 0 or not np.isfinite(elevation).any():
            logger.warning(f"Elevation data empty or invalid for bounds: {self.bounds}")
            raise ElevationDataError(
                "Elevation data is empty or invalid for the given area."
            )

        masked = np.ma.masked_where(
            ~np.isfinite(elevation) | (elevation <= -32768), elevation
        )
        min_elev = max(-500, float(masked.min()))
        max_elev = min(10000, float(masked.max()))
        return {"min": min_elev, "max": max_elev}
