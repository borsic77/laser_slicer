import logging
from pathlib import Path

import rasterio

from core.services.bathymetry_service import BathymetryFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)


def check_etopo_fetch():
    # La Gomera coordinates
    # Center: 28.11924, -17.22575
    # Bounds: ~1414m wide.
    # +/- 0.01 deg is ~2km. Safe.
    lat = 28.11924
    lon = -17.22575
    delta = 0.01
    bounds = (lon - delta, lat - delta, lon + delta, lat + delta)

    print(f"Checking ETOPO fetch for bounds: {bounds}")

    # Try 15s URL tile N30W030
    test_url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/15s/15s_bed_elev_gtif/ETOPO_2022_v1_15s_N30W030_bed.tif"
    print(f"Using Test URL: {test_url}")
    BathymetryFetcher.ETOPO_URL = test_url

    # Use a local cache dir for test
    cache_dir = Path("tmp/test_cache_15s_tile")
    fetcher = BathymetryFetcher(cache_dir=cache_dir)

    try:
        data = fetcher.fetch_elevation_for_bounds(bounds)
        print(f"Fetched Data Shape: {data.shape}")
        print(f"Min: {data.min()}")
        print(f"Max: {data.max()}")
        print(f"Mean: {data.mean()}")

        if data.min() > 0:
            print("WARNING: All data is POSITIVE! Pass ETOPO sees only land here??")
        else:
            print("SUCCESS: Data contains negative values (ocean).")

    except Exception as e:
        print(f"FETCH FAILED: {e}")


if __name__ == "__main__":
    check_etopo_fetch()
