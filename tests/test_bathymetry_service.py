import os
from pathlib import Path

import django
import numpy as np

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.services.bathymetry_service import BathymetryFetcher


def test_fetch_bathymetry():
    fetcher = BathymetryFetcher()

    # 1. Test Europe (Should use EMODnet)
    # Calera, La Gomera: -17.3, 28.1
    bounds_eu = (-17.340, 28.090, -17.330, 28.100)
    print(f"\n--- Testing Europe (EMODnet) ---")
    try:
        data_eu = fetcher.fetch_elevation_for_bounds(bounds_eu)
        print(f"Europe Fetch successful! Shape: {data_eu.shape}")
        print(f"Min Elevation: {np.nanmin(data_eu):.2f}m")
        assert data_eu.size > 0
    except Exception as e:
        print(f"Europe Fetch Failed: {e}")
        raise

    # 2. Test Global (Should use GEBCO)
    # Mariana Trench: 142.2, 11.3
    bounds_global = (142.15, 11.30, 142.25, 11.40)
    print(f"\n--- Testing Global (GEBCO) ---")
    try:
        data_global = fetcher.fetch_elevation_for_bounds(bounds_global)
        print(f"Global Fetch successful! Shape: {data_global.shape}")
        print(f"Min Elevation: {np.nanmin(data_global):.2f}m")
        assert data_global.size > 0
        assert np.nanmin(data_global) < -5000  # Deep ocean
    except Exception as e:
        print(f"Global Fetch Failed: {e}")
        raise

    print("\nALL INFRASTRUCTURE TESTS PASSED!")


if __name__ == "__main__":
    test_fetch_bathymetry()
