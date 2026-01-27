import requests


def list_collections():
    url = "https://data.geo.admin.ch/api/stac/v0.9/collections"
    resp = requests.get(url)
    data = resp.json()
    print(f"Found {len(data['collections'])} collections.")
    for c in data["collections"]:
        print(f"ID: {c['id']}")


if __name__ == "__main__":
    list_collections()
