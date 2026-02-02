import os
import sys

import django
import numpy as np
from shapely.geometry import shape

# Setup Django environment
sys.path.append("/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.services.contour_generator import ContourSlicingJob


def test_location(name, bounds, center):
    print(f"\n--- Testing {name} ---")
    job = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=50.0,
        num_layers=10,
        simplify=0.1,
        substrate_size_mm=400,
        layer_thickness_mm=3,
        center=center,
        smoothing=0,
        min_area=0.0,
        min_feature_width_mm=0,
        include_bathymetry=True,
    )

    contours = job.run()
    print(f"Total Layers: {len(contours)}")

    elevs = [c["elevation"] for c in contours]
    print(f"Elevations: {elevs}")

    neg = [e for e in elevs if e < 0]
    print(f"Bathy Layers (<0): {len(neg)}")

    if len(contours) > 0:
        base = contours[0]
        print(
            f"Base Elevation: {base['elevation']}m, Area: {shape(base['geometry']).area:.4f} units^2"
        )


if __name__ == "__main__":
    # 1. Coastal (La Gomera)
    test_location(
        "La Gomera (Coastal)",
        (-17.35311, 28.07498, -17.32446, 28.10139),
        (-17.33878, 28.08818),
    )

    # 2. Deep Ocean (Near Madeira)
    # 32.5, -16.5
    test_location("Deep Ocean", (-16.55, 32.45, -16.45, 32.55), (-16.5, 32.5))

    # 3. Flat Land (Zurich)
    test_location("Landlocked (Zurich)", (8.5, 47.3, 8.6, 47.4), (8.55, 47.35))
