import logging
import os
from pathlib import Path

import numpy as np
import rasterio
import requests
from django.conf import settings
from filelock import FileLock
from rasterio.io import MemoryFile
from rasterio.windows import Window, from_bounds

try:
    from core.utils.cache_ops import enforce_cache_limit, touch_cache_file
except ImportError:  # pragma: no cover

    def touch_cache_file(path):
        pass

    def enforce_cache_limit(cache_dir, max_bytes):
        pass


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
        return (-45 <= lon_min <= lon_max <= 45) and (25 <= lat_min <= lat_max <= 80)

    def fetch_elevation_for_bounds(
        self, bounds: tuple[float, float, float, float]
    ) -> np.ndarray:
        """Fetch bathymetry data for the given bounds."""
        lon_min, lat_min, lon_max, lat_max = bounds

        cache_key = (
            f"bathy_highres_{lon_min:.5f}_{lat_min:.5f}_{lon_max:.5f}_{lat_max:.5f}.tif"
        )
        cache_path = self.cache_dir / cache_key
        lock_path = str(cache_path) + ".lock"

        with FileLock(lock_path, timeout=60):
            if cache_path.exists():
                logger.info(f"Loading bathymetry from cache: {cache_path}")
                try:
                    with rasterio.open(cache_path) as src:
                        data = src.read(1)
                    touch_cache_file(cache_path)
                    return data
                except Exception as e:
                    logger.warning(f"Failed to read cache {cache_path}: {e}. Removing.")
                    cache_path.unlink(missing_ok=True)

            sources = []
            if self.is_in_europe(bounds):
                wcs_url = (
                    f"{self.EMODNET_WCS_URL}?service=WCS&version=2.0.1&request=GetCoverage"
                    f"&CoverageId=emodnet__mean_2022&format=image/tiff"
                    f"&subset=Lat({lat_min:.6f},{lat_max:.6f})&subset=Long({lon_min:.6f},{lon_max:.6f})"
                )
                sources.append(("EMODnet Europe (115m)", wcs_url))

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
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()

                        with MemoryFile(response.content) as memfile:
                            with memfile.open() as src:
                                data = src.read(1)
                                nodata = src.nodata
                                win_transform = src.transform
                    else:
                        with rasterio.open(url) as src:
                            window = from_bounds(
                                lon_min, lat_min, lon_max, lat_max, src.transform
                            )
                            if window.width < 1 or window.height < 1:
                                window = Window(
                                    int(window.col_off),
                                    int(window.row_off),
                                    max(1, int(round(window.width))),
                                    max(1, int(round(window.height))),
                                )
                            data = src.read(1, window=window)
                            nodata = src.nodata
                            win_transform = src.window_transform(window)

                    if data is not None and data.size > 0:
                        if nodata is not None and not np.isnan(nodata):
                            if not np.issubdtype(data.dtype, np.floating):
                                data = data.astype(np.float32)
                            data[data == nodata] = np.nan
                        self._save_to_cache(data, cache_path, win_transform)
                        return data
                except Exception as e:
                    logger.error(f"Failed to fetch from {name}: {e}")
                    continue

        raise RuntimeError("All bathymetry sources failed.")

    def _save_to_cache(self, data, path: Path, transform):
        """Save the cropped data to a local GeoTIFF for caching."""
        tmp_path = path.with_suffix(".tmp.tif")
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
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(data, 1)
            os.replace(tmp_path, path)
            logger.info(f"Saved bathymetry cache to {path}")
            enforce_cache_limit(
                self.cache_dir,
                int(getattr(settings, "BATHY_CACHE_MAX_MB", 512)) * 1024 * 1024,
            )
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            logger.warning(f"Failed to save cache file {path}: {e}")
