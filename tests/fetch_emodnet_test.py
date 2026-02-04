import sys

import requests


def test_emodnet_wms():
    print("Testing EMODnet Bathymetry WMS...")

    base_url = "https://ows.emodnet-bathymetry.eu/wms"
    params = {"service": "WMS", "request": "GetCapabilities"}

    try:
        print(f"Querying: {base_url} ...")
        response = requests.get(base_url, params=params, timeout=10)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            if (
                "WMS_Capabilities" in response.text
                or "WMT_MS_Capabilities" in response.text
            ):
                print("Content: Valid WMS Capabilities XML found.")
                print("SUCCESS: EMODnet Bathymetry WMS is accessible.")
            else:
                print("WARNING: Response 200 but XML content check failed.")
                print(f"Preview: {response.text[:200]}")
        else:
            print(f"FAILURE: Unexpected status code {response.status_code}")
            sys.exit(1)

    except Exception as e:
        print(f"FAILURE: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_emodnet_wms()
