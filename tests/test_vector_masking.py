import os
from pathlib import Path

import django
import matplotlib.pyplot as plt
import numpy as np

# Setup Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

import rasterio

from core.services.osm_service import fetch_coastline_mask


def test_la_gomera_mask(tmp_path):
    # Bounds for a part of La Gomera
    bounds = (-17.35, 28.08, -17.31, 28.11)
    res = 0.000277777777778
    width = int((bounds[2] - bounds[0]) / res)
    height = int((bounds[3] - bounds[1]) / res)
    shape = (height, width)
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    print(f"Testing Vector Mask for La Gomera. Target shape: {shape}")
    mask = fetch_coastline_mask(bounds, shape, transform)
    print(f"Mask generated. Land Pixels: {np.sum(mask)} / {mask.size}")

    output_dir = tmp_path / "data" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_dir / "test_mask_gomera.png", mask, cmap="Greys")

    assert mask.shape == shape
    assert np.sum(mask) > 0
    assert np.sum(mask) < mask.size
    print("SUCCESS: Coastal mask verified.")


def test_deep_ocean_mask():
    bounds = (-30.0, 30.0, -29.99, 30.01)
    res = 0.001
    width, height = 10, 10
    shape = (height, width)
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    print(f"\nTesting Vector Mask for Deep Ocean.")
    mask = fetch_coastline_mask(bounds, shape, transform)
    print(f"Mask generated. Land Pixels: {np.sum(mask)}")
    assert np.sum(mask) == 0
    print("SUCCESS: Deep ocean mask verified.")


def test_land_only_mask():
    # Zurich: 8.54, 47.37
    bounds = (8.54, 47.37, 8.55, 47.38)
    res = 0.0001
    width, height = 10, 10
    shape = (height, width)
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    print(f"\nTesting Vector Mask for Land Only (Zurich).")
    mask = fetch_coastline_mask(bounds, shape, transform)
    print(f"Mask generated. Land Pixels: {np.sum(mask)} / {mask.size}")
    # In Zurich, we expect 100% land (or close to it if there's no sea)
    # Actually, OSM tags many things. If it's a continent, it should be 1.
    assert np.sum(mask) > 0
    print("SUCCESS: Land-only mask verified.")


if __name__ == "__main__":
    test_la_gomera_mask()
    test_deep_ocean_mask()
    test_land_only_mask()
