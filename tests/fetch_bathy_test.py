import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.session import AWSSession
from rasterio.windows import from_bounds


def fetch_bathymetry_etopo(min_lon, min_lat, max_lon, max_lat):
    """
    Fetches bathymetry data from NOAA ETOPO 2022 via GeoTIFF using rasterio (VSICURL).
    Uses the 60 arc-second global dataset.
    """
    # ETOPO 2022 60s Bed Elevation (Global) GeoTIFF URL
    url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif"

    # Rasterio's CP_NO_WORK_ON_VIRTUAL_MEM_IO helps with some VSI curl operations
    # but usually default env is fine.

    print(f"Opening remote GeoTIFF: {url}")
    print(f"Target bounds: {min_lon}, {min_lat}, {max_lon}, {max_lat}")

    try:
        with rasterio.open(url) as src:
            # Create a window for the requested bounds
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)

            # Read the data in that window
            print(f"Reading window: {window}")
            data = src.read(1, window=window)

            # handle nodata
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)

            return data

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def plot_preview(data):
    if data is None or data.size == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap="terrain", origin="upper")
    plt.colorbar(label="Elevation (m)")
    plt.title("Bathymetry Preview (ETOPO 2022 GeoTIFF)")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")

    output_image = "bathymetry_preview.png"
    plt.savefig(output_image)
    print(f"Preview saved to {output_image}")

    print(f"Data shape: {data.shape}")
    print(f"Min elevation: {np.min(data)} m")
    print(f"Max elevation: {np.max(data)} m")


if __name__ == "__main__":
    # Test coordinates: Mariana Trench approx
    bounds = (142.1, 11.1, 142.5, 11.5)

    print("Starting bathymetry fetch test (ETOPO GeoTIFF)...")
    data = fetch_bathymetry_etopo(*bounds)

    if data is not None:
        plot_preview(data)
        print("Test completed successfully.")
    else:
        print("Test failed.")
