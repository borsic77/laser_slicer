import os
import sys
from pathlib import Path

import django
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, shape

# Setup Django environment
sys.path.append("/app")  # Assuming running in Docker
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from core.services.contour_generator import ContourSlicingJob


def calculate_blockiness(geometry):
    """
    Calculate the percentage of the geometry's boundary that consists of
    long (>100m) axis-aligned segments.
    """
    if geometry is None or geometry.is_empty:
        return 0.0

    long_aligned_length = 0.0
    total_length = geometry.length

    # Iterate over all polygons in the MultiPolygon
    geoms = geometry.geoms if hasattr(geometry, "geoms") else [geometry]

    for poly in geoms:
        # Check exterior and interiors
        rings = [poly.exterior] + list(poly.interiors)
        for ring in rings:
            coords = list(ring.coords)
            for i in range(len(coords) - 1):
                p1 = np.array(coords[i])
                p2 = np.array(coords[i + 1])

                segment_vector = p2 - p1
                length = np.linalg.norm(segment_vector)

                # If geometry represents ~5km, and total width is ~0.4m (substrate size).
                # Scale is approx 1/12500.
                # 100 meters -> 0.008 units.
                # Let's set the threshold to 0.005 (5mm) to be sensitive.
                if length < 0.005:
                    continue

                # Check axis alignment
                # Horizontal: dy is small
                # Vertical: dx is small
                dx = abs(segment_vector[0])
                dy = abs(segment_vector[1])

                # Tolerance for "aligned" (e.g. 1 degree is roughly 1/60 slope)
                # But here we are in projected coordinates (meters) usually?
                # Wait, ContourSlicingJob returns projected geometry in "mm" usually if scaled?
                # No, the raw contours are in meters if we intercept them early,
                # but the job returns them scaled to substrate.
                # Actually, let's look at the job output.
                # It returns a list of dicts with 'geometry' (geojson).
                # The job scales to substrate size (mm).
                # If substrate is 400mm, and area is 5km, then 100m ~ 8mm.
                # Let's adjust threshold to "2% of total width" or similar relative metric?
                # OR, we can try to intercept the geometry BEFORE scaling.
                # But `run()` does everything.
                # Let's stick to the output.
                # If width is ~5000m and substrate is 400mm. Scale is 0.08 mm/m.
                # 100m block -> 8mm line.

                is_horizontal = dy < (length * 0.05)
                is_vertical = dx < (length * 0.05)

                # NEW: Filter out bounding box edges (perfectly straight and at the limits)
                # Bounds are roughly (-0.2 to 0.2)
                # Let's count only segments that are NOT on the outer frame
                margin = 0.001
                if (
                    abs(p1[0]) > 0.199
                    or abs(p2[0]) > 0.199
                    or abs(p1[1]) > 0.199
                    or abs(p2[1]) > 0.199
                ):
                    # This segment is likely on the bbox boundary
                    continue

                if is_horizontal or is_vertical:
                    long_aligned_length += length

    if total_length == 0:
        return 0.0

    return (long_aligned_length / total_length) * 100.0


def run_test():
    print("Running Blockiness Test against 'La Calera, La Gomera'...")

    # Coordinates from previous logs: Center: 28.09386, -17.33966
    # Width/Height ~ 5.6km
    # Bounds roughly:
    center_lat = 28.09386
    center_lon = -17.33966
    delta_deg = 0.025  # approx 2.5km radius

    bounds = (
        center_lon - delta_deg,
        center_lat - delta_deg,
        center_lon + delta_deg,
        center_lat + delta_deg,
    )

    job = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=100.0,  # Large layers to just get the coastline
        num_layers=1,  # We only care about the base layer (coastline)
        simplify=0.0,  # No simplification to see raw blocks
        substrate_size_mm=400,
        layer_thickness_mm=5,
        center=(center_lon, center_lat),
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_bathymetry=True,
    )

    # We need to monkey-patch report because it's required arg? No, it defaults to None in run() definition?
    # run(self, progress_callback=None)

    print("Executing slicing job...")
    contours = job.run()

    # Find the 0m contour (or lowest positive if 0 is missing, but should be 0)
    # Actually, with bathymetry, the lowest layer starts at deep ocean.
    # The "Coastline" is effectively the 0m contour.
    # But `contours` list depends on `num_layers` and range.
    # If we set min/max range?
    # The job calculates min/max from data.
    # If we want specifically the 0m contour, we might need to inspect all of them.

    coastline_contour = None
    for c in contours:
        # We look for elevation close to 0.
        # Note: if bathymetry is included, range is -500 to +500.
        # 0m layer should exist if we are lucky with spacing.
        # Or we can just sum up the blockiness of ALL layers.
        # If the failure is "blocky coastline", the 0m transition is key.
        # But wait, if slicing, we might not get exactly 0m.
        # However, the user complains about the "coastline".
        # Let's check the layer with elevation closest to 0.
        if abs(c["elevation"]) < 50:  # arbitrary closeness
            coastline_contour = c
            # Keep looking for better match?
            pass

    if not coastline_contour:
        print("Warning: Could not find a contour near 0m. Using the middle one.")
        coastline_contour = contours[len(contours) // 2]

    geom = shape(coastline_contour["geometry"])
    print(f"Analyzing contour at {coastline_contour['elevation']}m...")
    print(f"DEBUG: Geometry Bounds (minx, miny, maxx, maxy): {geom.bounds}")
    print(f"DEBUG: Total Length: {geom.length}")

    score = calculate_blockiness(geom)
    print(f"Blockiness Score (LAAS %): {score:.2f}%")

    # Final visual check: Save a plot of the coastline
    output_dir = Path("/app/data/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, color="black", linewidth=0.5)
    else:
        x, y = geom.exterior.xy
        ax.plot(x, y, color="black", linewidth=0.5)
    ax.set_title(f"Coastline (0m) - Blockiness: {score:.2f}%")
    ax.set_aspect("equal")
    plt.savefig(output_dir / "coastline_smoothness_check.png")
    print(f"Saved visual check to {output_dir / 'coastline_smoothness_check.png'}")

    if score > 20.0:
        print("FAIL: Coastline is too blocky.")
    elif score > 5.0:
        print("WARNING: Moderate blockiness.")
    else:
        print("PASS: Coastline is smooth.")


if __name__ == "__main__":
    run_test()
