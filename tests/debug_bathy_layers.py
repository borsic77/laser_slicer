import os
import sys

import django
import numpy as np

# Setup Django environment
sys.path.append("/app")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.services.contour_generator import ContourSlicingJob


def debug_run():
    # Use the coordinates from the user's logs
    bounds = (-17.35311, 28.07498, -17.32446, 28.10139)
    center = (-17.33878, 28.08818)

    print(f"Running Debug Job for bounds: {bounds}")

    job = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=50.0,
        num_layers=10,
        simplify=0.1,
        substrate_size_mm=400,
        layer_thickness_mm=3,
        center=center,
        smoothing=50,
        min_area=10.0,
        min_feature_width_mm=1.0,
        include_bathymetry=True,
    )

    contours = job.run()

    print(f"\nResults: {len(contours)} layers generated.")
    for i, c in enumerate(contours):
        elev = c["elevation"]
        geom = c["geometry"]
        # Compute area of the geometry in scaled units (mm^2)
        from shapely.geometry import shape

        s = shape(geom)
        area_mm2 = s.area * (1000.0**2)  # Wait, scaling might be different
        # Actually, the job scales it to substrate size.
        # If substrate is 400mm, total area is ~160,000 mm^2.
        print(f"Layer {i}: Elevation = {elev:7.2f}m, Area = {s.area:.6f} units^2")

    # Check if we have negative elevations
    neg_layers = [c for c in contours if c["elevation"] < 0]
    print(f"\nNegative (Bathymetry) Layers: {len(neg_layers)}")
    if not neg_layers:
        print("CRITICAL: No bathymetry layers found!")
    else:
        for c in neg_layers:
            print(f"  Found: {c['elevation']}m")


if __name__ == "__main__":
    debug_run()
