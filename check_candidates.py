import requests

candidates = [
    "ch.swisstopo.swissalti3d",
    "ch.swisstopo.swissalti3d-2m",
    "ch.swisstopo.swissalti3d-2m-grid",
    "ch.swisstopo.swissalti3d-0.5m",
]


def check_candidates():
    for cid in candidates:
        url = f"https://data.geo.admin.ch/api/stac/v0.9/collections/{cid}"
        resp = requests.get(url)
        if resp.status_code == 200:
            print(f"FOUND: {cid}")
            # Identify asset types
            data = resp.json()
            print(f"Title: {data.get('title')}")
        else:
            print(f"Not Found: {cid}")


if __name__ == "__main__":
    check_candidates()
