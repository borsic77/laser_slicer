import logging
import os
from pathlib import Path

import numpy as np
import rasterio
from django.conf import settings
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)


class BathymetryFetcher:
    """Service to fetch global bathymetry data (ETOPO 2022)."""

    # ETOPO 2022 60s Bed Elevation (Global) GeoTIFF URL
    ETOPO_URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.BATHY_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_elevation_for_bounds(
        self, bounds: tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Fetch bathymetry/elevation data from ETOPO 2022 for the given bounds.

        Args:
            bounds: Tuple (lon_min, lat_min, lon_max, lat_max) in WGS84.

        Returns:
            np.ndarray: Elevation data (negative for ocean depth).
        """
        lon_min, lat_min, lon_max, lat_max = bounds

        # Simple cache filename based on bounds (precision 4 decimals -> ~11m)
        cache_key = f"etopo_{lon_min:.4f}_{lat_min:.4f}_{lon_max:.4f}_{lat_max:.4f}.tif"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            logger.info(f"Loading bathymetry from cache: {cache_path}")
            try:
                with rasterio.open(cache_path) as src:
                    return src.read(1)
            except Exception as e:
                logger.warning(
                    f"Failed to read cache {cache_path}: {e}. Removing and re-fetching."
                )
                os.remove(cache_path)

        logger.info(f"Fetching bathymetry from ETOPO 2022 (VSICURL): {self.ETOPO_URL}")
        try:
            with rasterio.open(self.ETOPO_URL) as src:
                window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
                logger.debug(f"Reading window: {window} for bounds {bounds}")

                # Expand window slightly to ensure coverage if needed, but from_bounds usually handles it.
                # ETOPO is pixel-is-area, so nearest alignment.

                data = src.read(1, window=window)

                # Check for nodata
                if src.nodata is not None:
                    # ETOPO usually doesn't have nodata for global, but standard practice
                    data = np.where(data == src.nodata, np.nan, data)

                # Save to cache
                self._save_to_cache(data, cache_path, bounds, src.transform, window)

                return data

        except Exception as e:
            logger.error(f"Failed to fetch bathymetry from ETOPO: {e}")
            raise

    def _save_to_cache(self, data, path, bounds, src_transform, window):
        """Save the cropped data to a local GeoTIFF for caching."""
        try:
            # We need to compute the transform for the new window
            # src.window_transform(window) is the way
            with (
                rasterio.open(self.ETOPO_URL) as src
            ):  # Re-open just to be safe or calc manually if window implies transform
                win_transform = src.window_transform(window)

            profile = {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "dtype": data.dtype,
                "crs": "EPSG:4326",
                "transform": win_transform,
            }

            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            logger.info(f"Saved bathymetry cache to {path}")

        except Exception as e:
            logger.warning(f"Failed to save cache file {path}: {e}")
