import logging

import numpy as np

from core.services.contour_generator import ContourSlicingJob

# Configure logger to see output
logging.basicConfig(level=logging.DEBUG)


def test_no_bathymetry_clamping():
    # La Gomera coordinates from screenshot
    # Center: 28.11924, -17.22575
    # Width ~ 1414m
    # Let's define a small bounds around this center
    lat = 28.11924
    lon = -17.22575
    delta = 0.01  # approx 1km
    bounds = (lon - delta, lat - delta, lon + delta, lat + delta)
    center = (lon, lat)

    print("Initialize Job with include_bathymetry=False")
    job = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=50.0,
        num_layers=5,
        simplify=0.5,
        substrate_size_mm=400,
        layer_thickness_mm=5,
        center=center,
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_bathymetry=False,
    )

    print("Running job...")
    contours = job.run()

    # Check elevations
    elevations = [c["elevation"] for c in contours]
    print(f"Generated {len(contours)} contours.")
    print(f"Elevations: {elevations}")

    min_elev = min(elevations) if elevations else 0
    print(f"Minimum Elevation: {min_elev}")

    if min_elev < 0:
        print(
            "FAIL: Found negative 'elevation' contours even though bathymetry was disabled!"
        )
        exit(1)
    else:
        print("pass: All contours are >= 0")

    # Also test with bathymetry enabled for comparison
    print("\nInitialize Job with include_bathymetry=True")
    job_bathy = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=50.0,
        num_layers=5,
        simplify=0.5,
        substrate_size_mm=400,
        layer_thickness_mm=5,
        center=center,
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_bathymetry=True,
    )
    # We might fail to fetch ETOPO if not configured/mocked, but let's see.
    # If it fails to fetch, it falls back to SRTM, so we might just get SRTM data.
    # But SRTM should have voids.
    try:
        contours_bathy = job_bathy.run()
        elevations_bathy = [c["elevation"] for c in contours_bathy]
        print(f"Elevations with Bathy: {elevations_bathy}")
    except Exception as e:
        print(f"Bathy run failed (expected if network/files missing): {e}")


if __name__ == "__main__":
    test_no_bathymetry_clamping()
