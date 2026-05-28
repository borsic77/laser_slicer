# Laser Slicer – Laser‑Cuttable Contour Map Generator

Turn any point on Earth into a **stack of precisely‑aligned SVG layers ready for laser cutting**.  
Laser Slicer downloads SRTM elevation data, OSM features like roads and waterways, slices contours at the interval you choose, and gives you:

Live site: [laserslicer.legradic.ch](https://laserslicer.legradic.ch)
![Screenshot of the Laser Slicer interface](screenshot.png)

* A **ZIP archive** of SVG files (one per contour layer)  
* A **browser 3‑D preview** of the stacked model
* A **Dockerised, reproducible workflow** that runs the heavy lifting in Celery workers
* **Fixed elevation** tool for clean coastlines and lakes
* **Bathymetry support** (Ocean Data) using global ETOPO datasets
* **SRTM noise filtering** for troublesome areas (e.g. Boston)



---

## Features

| Front‑end (Vite + React)                | Back‑end (Django 5 + Celery)          |
|-----------------------------------------|---------------------------------------|
| **Leaflet map** – pick any address or drop a pin | **Geocoding** through Nominatim |
| Auto geolocation start | `/api/elevation` single‑point query |
| Parameter panel – slice height, base height, simplification | Fetch & clip **SRTM 30 m** DEM |
| Minimum area/width filters & optional fixed elevation (lake) | **SwissALTI3D 2m** (automatic fallback) |
| Toggle roads/buildings/waterways overlay | Robust SRTM cleaning & outlier filtering |
| Toggle roads/buildings/waterways overlay | Fetch OSM roads, buildings & waterways |
| **Slice** button launches background job | Contour generation with GDAL / Matplotlib |
| Live **Three.js** 3‑D preview of layers | **Parallel Processing** for faster slicing |
| **Download SVGs** – packaged as a ZIP | Optional simplification with Shapely |
| **Download SVGs** – packaged as a ZIP | SVG export via `svgwrite` |
| Hot‑reload dev server (`npm run dev`) | **Celery + Redis** task queue |
| | Single‑command **Docker** deploy |

---

## High-Level Architecture

The application defines a clear pipeline for generating laser-cuttable maps:

1.  **Frontend (React + Vite)**: User selects an area and configures slicing parameters (layer height, substrate size).
2.  **API (Django)**: Receives the job request and creates a record in the database (`ContourJob`).
3.  **Task Queue (Celery + Redis)**: The heavy processing is offloaded to background workers.
4.  **Worker Logic**:
    *   **Elevation Fetch**: Downloads SRTM 30m DEM tiles or high-res SwissALTI3D data.
    *   **OSM Geometry**: Fetches roads, buildings, and waterways/waterbodies from Overpass/OSM.
    *   **Contour Generation**: Uses `matplotlib` (marching squares) or `gdal` to create vector contours from the DEM.
    *   **Geometry Processing**: Uses `shapely` for cleaning, simplifying, orienting, and subtracting layers (e.g., carving out water).
5.  **Result**: The final layers are optimized and exported as a ZIP of **SVG** files, stored in the media volume.

---

## Quick Start (Docker)

```bash
# 1. Clone the repo
git clone https://github.com/borsic77/laser_slicer.git
cd laser_slicer

# 2. Configure environment files
# .env       – local-dev settings (VITE_API_URL, optional Django overrides)
cp env.example .env

# .env.docker – Docker-specific settings (Postgres credentials, USER_AGENT).
# docker-compose.yml reads this file; without it the db/backend/celery
# containers will fail to start.
cp env.docker.example .env.docker
# Edit USER_AGENT in .env.docker to identify your instance (OSM policy).

# 3. Fire everything up
docker compose up --build -d
```

Visit **http://localhost:5173** in your browser.
The front‑end talks to the Django API at **http://localhost:8000**.
The API enforces CSRF protection. Ensure your `.env` defines matching
`DJANGO_CORS_ALLOWED_ORIGINS` and `DJANGO_CSRF_TRUSTED_ORIGINS`
so the browser can send the `csrftoken` cookie with each request.



## Local Development (without Docker)

| Requirement | Version |
|-------------|---------|
| Python      | 3.10 +  |
| Node        | 18 +    |
| Postgres    | 16      |
| Redis       | 7       |
| GDAL        | 3.6 +   |

See the **[Developer Startup Guide](dev_startup.md)** for detailed, step-by-step setup instructions to run the project locally without Docker.


---

## Project Layout

```
.
├── core/               # Django project
├── frontend/           # Vite + React app
├── data/srtm_cache/    # DEM cache (mounted volume)
├── docker-compose*.yml
└── README.md
```

---

## API Endpoints

| Method | Endpoint                        | Description                       |
|--------|----------------------------------|-----------------------------------|
| `POST` | `/api/contours/`                | Start slicing task with parameters |
| `GET`  | `/api/contours/<uuid>/`         | Check status and result of a task |
| `GET`  | `/api/contours/<uuid>/download/`| Download resulting ZIP file       |
| `GET`  | `/api/elevation-range/`         | Get min/max elevation for a given bounding box |
| `GET`  | `/api/elevation`                | Sample elevation at a lat/lon |
| `GET`  | `/api/status/`                  | Basic health check/status         |



---

## Contributing

1. Fork the project & create a feature branch (`git checkout -b feature/my_feature`)
2. Commit your changes with conventional commits
3. Run `pre‑commit run --all-files` and ensure tests pass (`pytest`)
4. Open a Pull Request 😊

---

## License

Distributed under the **MIT License**.  
See [`LICENCE`](LICENCE) for more information.

---

## Author

Created by **Boris Legradic**  
Feel free to reach out via [GitHub](https://github.com/borsic77) or [email](mailto:boris@legradic.ch)

---

## Acknowledgements

* NASA / NGA – SRTM data  
* SwissALTI3D - Elevation data for Switzerland
* OpenStreetMap & Nominatim  
