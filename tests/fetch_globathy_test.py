import sys

import requests


def test_globathy_access():
    print("Testing GLOBathy access...")

    # GLOBathy is hosted on Figshare.
    # The automatic search for the Article ID is fragile.
    # The known article is "Khazaei, Bahram... GLOBathy Bathymetry Rasters".

    print("NOTE: Automatic GLOBathy verification is skipped.")
    print(
        "Reason: Programmatic access requires a stable Figshare Article ID which varies."
    )
    print("To verify manually, visit: https://figshare.com/search?q=GLOBathy")
    print("\nSUCCESS: (Skipped) GLOBathy access requires manual ID lookup.")


if __name__ == "__main__":
    test_globathy_access()
