import logging

import numpy as np

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.dem import clean_srtm_dem, mosaic_and_crop

logger = logging.getLogger(__name__)


class ElevationDataError(ValueError):
    """Raised when elevation data is missing or invalid within a bounding box."""

    pass


class ElevationRangeJob:
    """Class to compute the elevation range within a specified bounding box.
    This class is responsible for downloading elevation data and calculating
    the minimum and maximum elevation values.
    """

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
        tile_paths = download_srtm_tiles_for_bounds(self.bounds)
        elevation, _ = mosaic_and_crop(tile_paths, self.bounds)
        elevation = clean_srtm_dem(elevation)
        # elevation = robust_local_outlier_mask(elevation)
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
