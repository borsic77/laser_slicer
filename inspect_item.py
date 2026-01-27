import json

import requests


def inspect_item():
    url = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
    # Get just one item
    resp = requests.get(url, params={"limit": 1})
    data = resp.json()
    if data["features"]:
        item = data["features"][0]
        print(json.dumps(item["assets"], indent=2))
        print("\nItem ID:", item["id"])
    else:
        print("No items found.")


if __name__ == "__main__":
    inspect_item()
