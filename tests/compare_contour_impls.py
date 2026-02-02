import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import shape

# Add project root to path BEFORE importing core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

django.setup()

from core.utils.contour_ops import generate_contours


def create_synthetic_dem(size=100):
    """Create a simple synthetic DEM with a known shape (gaussian hill)."""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / 2) * 100  # Gaussian hill, peak 100m

    # Create a dummy transform (1 unit per pixel)
    transform = rasterio.Affine.translation(0, 0) * rasterio.Affine.scale(1, 1)
    return Z, transform


def test_current_implementation():
    print("Testing CURRENT implementation (Matplotlib)...")
    dem, transform = create_synthetic_dem()
    masked_dem = np.ma.masked_array(dem, mask=False)

    # Generate contours at 10m intervals
    contours = generate_contours(
        masked_dem,
        dem,
        transform,
        interval=10,
        simplify=0,
        debug_image_path="tests/debug_output",
    )

    print(f"Generated {len(contours)} contour levels.")

    # Basic validation stats
    results = []
    for c in contours:
        geom = shape(c["geometry"])
        results.append(
            {
                "elevation": c["elevation"],
                "area": geom.area,
                "length": geom.length,
                "geom_type": geom.geom_type,
            }
        )
        print(f"Level {c['elevation']}: Area={geom.area:.2f}, Length={geom.length:.2f}")

    return results


if __name__ == "__main__":
    os.makedirs("tests/debug_output", exist_ok=True)
    results = test_current_implementation()

    # Save results to compare later
    import json

    with open("tests/baseline_contours.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBaseline results saved to tests/baseline_contours.json")
