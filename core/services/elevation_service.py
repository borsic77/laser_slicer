import logging

import numpy as np

from core.services.bathymetry_service import BathymetryFetcher
from core.utils.dem import (
    clean_srtm_dem,
    merge_srtm_and_etopo,
    mosaic_and_crop,
)
from core.utils.download_clip_elevation_tiles import (
    download_elevation_tiles_for_bounds,
)

logger = logging.getLogger(__name__)


class ElevationDataError(ValueError):
    """Raised when elevation data is missing or invalid within a bounding box."""

    pass


class ElevationRangeJob:
    """Class to compute the elevation range within a specified bounding box.
    This class is responsible for downloading elevation data and calculating
    the minimum and maximum elevation values.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        include_bathymetry: bool = False,
    ):
        """
        Initializes the elevation range job.
        Args:
            bounds: A tuple containing the bounding box coordinates (lon_min, lat_min, lon_max, lat_max).
            include_bathymetry: Whether to include deep ocean data.
        """
        self.bounds = bounds
        self.include_bathymetry = include_bathymetry

    def run(self) -> dict:
        """
        Computes the minimum and maximum elevation within the specified bounding box.
        Returns:
            A dictionary with 'min' and 'max' elevation values.
        """
        tile_paths = download_elevation_tiles_for_bounds(self.bounds)
        # Downsample by 10x for fast statistics (approx 20m res for Swiss data)
        # This drastically speeds up min/max calculation for map interactions
        elevation, transform = mosaic_and_crop(
            tile_paths, self.bounds, downsample_factor=10
        )

        if self.include_bathymetry:
            try:
                fetcher = BathymetryFetcher()
                etopo_data = fetcher.fetch_elevation_for_bounds(self.bounds)
                # We need to ensure etopo_data matches the downsampled grid?
                # merge_srtm_and_etopo resamples ETOPO to match the SRTM grid (elevation arg).
                # So this should work fine even with downsampled SRTM.
                elevation = merge_srtm_and_etopo(
                    elevation, etopo_data, bounds=self.bounds, transform=transform
                )
            except Exception as e:
                logger.warning(
                    f"Failed to fetch bathymetry for range calculation: {e}. Ignoring."
                )

        # Adjust cleaning and clamping based on bathymetry
        # If bathymetry is included, we expect values down to -11000.
        # If not, we clamp SRTM voids to something reasonable like -500 or 0,
        # but typically clean_srtm_dem handles voids.
        min_valid = -11000 if self.include_bathymetry else -500
        elevation = clean_srtm_dem(elevation, min_valid=min_valid)

        if elevation.size == 0 or not np.isfinite(elevation).any():
            logger.warning(f"Elevation data empty or invalid for bounds: {self.bounds}")
            raise ElevationDataError(
                "Elevation data is empty or invalid for the given area."
            )

        # Mask out voids that might still remain or be marker values
        masked = np.ma.masked_where(
            ~np.isfinite(elevation) | (elevation <= -32768), elevation
        )

        real_min = float(masked.min())
        real_max = float(masked.max())

        # Clamp only if NOT handling bathymetry, or ensure reasonable bounds
        # If bathymetry is ON, allow down to -11000. If OFF, clamp at -500.
        clamp_min = -11000 if self.include_bathymetry else -500

        min_elev = max(clamp_min, real_min)
        max_elev = min(10000, real_max)

        # If min > max (due to flat ocean at 0 and clamping?), safe fallback
        if min_elev > max_elev:
            min_elev = max_elev - 10

        return {"min": min_elev, "max": max_elev}
