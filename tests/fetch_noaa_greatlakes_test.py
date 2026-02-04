import sys

import requests


def test_noaa_wms():
    print("Testing NOAA Great Lakes Bathymetry WMS...")

    # Hosted by Esri Canada (fed by NOAA/NGDC)
    # Using HTTPS and User-Agent to avoid connection blocks
    base_url = "https://edumaps.esri.ca/ArcGIS/services/MapServices/GreatLakesBathymetry/MapServer/WMSServer"
    params = {"service": "WMS", "request": "GetCapabilities"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }

    try:
        print(f"Querying: {base_url} ...")
        response = requests.get(base_url, params=params, headers=headers, timeout=10)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            if (
                "WMS_Capabilities" in response.text
                or "WMT_MS_Capabilities" in response.text
            ):
                print("Content: Valid WMS Capabilities XML found.")
                print("SUCCESS: NOAA Great Lakes WMS is accessible.")
            else:
                print("WARNING: Response 200 but XML content check failed.")
                print(f"Preview: {response.text[:200]}")
        else:
            print(f"FAILURE: Unexpected status code {response.status_code}")
            sys.exit(1)

    except Exception as e:
        print(f"FAILURE: {e}")
        # Failure allows us to know we need another strategy, but we don't want to crash the whole CI chain if this specific optional source is down
        sys.exit(1)


if __name__ == "__main__":
    test_noaa_wms()
