import sys

import requests


def test_hydrolakes_access():
    print("Testing HydroLAKES access...")

    # URL derived from HydroSHEDS website structure
    # Official page: https://www.hydrosheds.org/products/hydrolakes
    url = "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"

    print(f"Checking URL: {url}")
    print("Method: HEAD (avoiding full download)")

    try:
        # Stream=True with HEAD is efficient
        response = requests.head(url, allow_redirects=True, timeout=10)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            content_length = response.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                print(f"Content-Length: {content_length} bytes ({size_mb:.2f} MB)")
            else:
                print("Content-Length header missing.")

            print("\nSUCCESS: HydroLAKES dataset is reachable.")
        else:
            print(f"\nFAILURE: Unexpected status code {response.status_code}")
            sys.exit(1)

    except Exception as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_hydrolakes_access()
