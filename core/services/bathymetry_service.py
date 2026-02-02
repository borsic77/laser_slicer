import io
import logging
import os
from pathlib import Path

import numpy as np
import rasterio
import requests
from django.conf import settings
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)


class BathymetryFetcher:
    """Coordinator to fetch bathymetry data from the best available source."""

    # GEBCO 2024 15s (Global) - Cloud Optimized GeoTIFF (via Natural Capital Project)
    GEBCO_URL = "https://storage.googleapis.com/natcap-data-cache/global/gebco/gebco_bathymetry_2024_global.tif"

    # ETOPO 2022 15s (Global) - Fallback
    ETOPO_URL = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/15s_bed_elev_gtif/ETOPO_2022_v1_15s_N90W180_bed.tif"

    # EMODnet WCS for high-res Europe (~115m)
    EMODNET_WCS_URL = "https://ows.emodnet-bathymetry.eu/wcs"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or settings.BATHY_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_in_europe(self, bounds: tuple[float, float, float, float]) -> bool:
        """Check if the bounds are within the EMODnet European coverage area."""
        lon_min, lat_min, lon_max, lat_max = bounds
        # Rough EMODnet extent: Macaronesia to Arctic, Mid-Atlantic to Caspian
        return (-45 <= lon_min <= 45) and (25 <= lat_min <= 80)

    def fetch_elevation_for_bounds(
        self, bounds: tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Fetch bathymetry data for the given bounds.
        Tries EMODnet for Europe (~115m), GEBCO 2024 (15s), then ETOPO 2022 (15s).
        """
        lon_min, lat_min, lon_max, lat_max = bounds

        # 1. Check Cache first
        cache_key = (
            f"bathy_highres_{lon_min:.5f}_{lat_min:.5f}_{lon_max:.5f}_{lat_max:.5f}.tif"
        )
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            logger.info(f"Loading bathymetry from cache: {cache_path}")
            try:
                with rasterio.open(cache_path) as src:
                    return src.read(1)
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_path}: {e}. Removing.")
                os.remove(cache_path)

        # 2. Tiered Fetching
        sources = []

        # Add EMODnet as top priority for Europe
        if self.is_in_europe(bounds):
            wcs_url = (
                f"{self.EMODNET_WCS_URL}?service=WCS&version=2.0.1&request=GetCoverage"
                f"&CoverageId=emodnet__mean_2022&format=image/tiff"
                f"&subset=Lat({lat_min:.6f},{lat_max:.6f})&subset=Long({lon_min:.6f},{lon_max:.6f})"
            )
            sources.append(("EMODnet Europe (115m)", wcs_url))

        # Standard high-res global sources
        sources.extend(
            [
                ("GEBCO 2024 (450m)", self.GEBCO_URL),
                ("ETOPO 2022 (450m)", self.ETOPO_URL),
            ]
        )

        for name, url in sources:
            logger.info(f"Attempting to fetch bathymetry from {name}: {url}")
            try:
                if "request=GetCoverage" in url:
                    # For WCS, use requests to handle dynamic TIFF blobs robustly
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    with MemoryFile(response.content) as memfile:
                        with memfile.open() as src:
                            data = src.read(1)
                            win_transform = src.transform
                else:
                    # For COGs, use rasterio's native VSICURL windows
                    with rasterio.open(url) as src:
                        window = from_bounds(
                            lon_min, lat_min, lon_max, lat_max, src.transform
                        )
                        data = src.read(1, window=window)
                        win_transform = src.window_transform(window)

                if data is not None:
                    # Handle nodata
                    # With MemoryFile, we can't easily check src.nodata here unless we are inside the context
                    # But most bathy sources are clean.

                    # Save to cache
                    self._save_to_cache(data, cache_path, win_transform)
                    return data
            except Exception as e:
                logger.error(f"Failed to fetch from {name}: {e}")
                continue

        raise RuntimeError("All bathymetry sources failed.")

    def _save_to_cache(self, data, path, transform):
        """Save the cropped data to a local GeoTIFF for caching."""
        try:
            profile = {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "dtype": data.dtype,
                "crs": "EPSG:4326",
                "transform": transform,
            }

            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            logger.info(f"Saved bathymetry cache to {path}")

        except Exception as e:
            logger.warning(f"Failed to save cache file {path}: {e}")
