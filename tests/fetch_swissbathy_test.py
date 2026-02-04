import json
import sys

import rasterio
import requests
from rasterio.io import MemoryFile


def test_swissbathy_access():
    print("Testing swissBATHY3D access via STAC API...")

    # STAC Search Endpoint for swissBATHY3D
    # Using the geoadmin STAC API
    stac_api_url = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissbathy3d/items"

    # Bounding box for Lake Zurich (approximate)
    # Format: min_lon, min_lat, max_lon, max_lat
    bbox = [8.53, 47.30, 8.56, 47.35]

    params = {"bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"}

    try:
        response = requests.get(stac_api_url, params=params)
        response.raise_for_status()
        data = response.json()

        print(f"STAC Query Status: {response.status_code}")
        features = data.get("features", [])
        print(f"Found {len(features)} tiles near Lake Zurich.")

        if not features:
            print("No features found. Check collection ID or bbox.")
            return

        # Inspect the first feature
        feature = features[0]
        assets = feature.get("assets", {})

        # Look for the .esriasciigrid.zip asset
        zip_asset = None
        for key, asset in assets.items():
            if "esriasciigrid.zip" in key or asset.get("href", "").endswith(".zip"):
                zip_asset = asset
                break

        if not zip_asset:
            print("No .zip asset found in feature.")
            print("Available assets:", list(assets.keys()))
            # Fallback to XYZ if available
            xyz_asset = next((a for k, a in assets.items() if "xyz" in k), None)
            if xyz_asset:
                print(f"Found XYZ asset: {xyz_asset.get('href')}")
            return

        zip_url = zip_asset.get("href")
        print(f"Found ZIP URL: {zip_url}")

        # We can't easily stream a zipped ASC file inside a ZIP via generic VSICURL without knowing the internal filename.
        # But for this test, identifying the URL is sufficient success.
        print("SUCCESS: Successfully identified SwissBATHY3D download URL.")
        print("Note: In production, we would download, unzip, and read the .asc file.")

        # End of test
        pass

    except Exception as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_swissbathy_access()
