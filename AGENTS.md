# AGENTS.md – Laser Slicer

Laser Slicer is a web application for selecting geographic regions, fetching elevation/bathymetry/OSM data, slicing terrain into stacked contour layers, previewing in 3D, and exporting laser-ready SVG files.

## 1. Repository Map

- `config/` — Django project configuration (`settings.py`, `urls.py`, `celery.py`, `wsgi.py`).
- `core/` — Django backend application:
  - `views.py`, `models.py`, `tasks.py`, `serializers.py` — REST API endpoints, `ContourJob` / `ElevationJob` / `SVGJob` models, and Celery tasks.
  - `services/` — High-level domain services:
    - `contour_generator.py` (`ContourSlicingJob` pipeline orchestration)
    - `elevation_service.py` (`ElevationRangeJob` min/max elevation calculation)
    - `bathymetry_service.py` (`BathymetryFetcher` for EMODnet / GEBCO / ETOPO)
    - `osm_service.py` (Overpass coastline/water masks)
    - `svg_zip_generator.py` (SVG layer generation and ZIP packaging)
  - `utils/` — Standalone geospatial, raster, and vector utilities (`dem.py`, `contour_ops.py`, `geometry_ops.py`, `parallel_ops.py`, `svg_export.py`, `osm_features.py`, `osm_tiles.py`, `swiss_topo_api.py`, `waterbody.py`, `geocoding.py`).
- `frontend/` — React 19 + TypeScript + Vite SPA:
  - `src/components/` — `Mapview.tsx` (Leaflet), `ContourPreview.tsx` (Three.js / R3F 3D preview), `Sidebar.tsx`, `Header.tsx`, `MapController.tsx`.
  - `src/hooks/` — `useElevationJob.ts`, `useSlicingJob.ts`, `useSelectionBounds.ts`.
  - `src/reducers/` — `slicerReducer.ts`.
  - `src/services/api.ts` — Backend API client.
- `tests/` — Backend `pytest` suite.
- `data/`, `media/` — Runtime tile caches (`srtm_cache`, `bathy_cache`, `alti3d_cache`, `osm_cache`) and exported job files (git-ignored).

## 2. Commands (Setup, Dev, Test, Lint)

```bash
# Python environment & dependencies
uv sync

# Backend lint & format
uvx ruff format .
uvx ruff check .

# Backend tests
.venv/bin/pytest -q

# Frontend setup, lint, test, build
npm --prefix frontend ci
npm --prefix frontend run lint
npm --prefix frontend test -- --run
npm --prefix frontend run build
```

For local development with full background task support, start Postgres and Redis (`docker compose up db redis -d`), run the Django dev server (`.venv/bin/python manage.py runserver`), Celery worker (`.venv/bin/celery -A config worker -l info`), and Vite dev server (`npm --prefix frontend run dev`).

## 3. Architecture & Coding Style

### Backend (Python / Django)
- **Layering**: Keep high-level orchestration in `core/services/` and pure geospatial/raster helpers in `core/utils/`. `core/utils/` must never import from `core/services/`.
- **Style**: Format and lint with `ruff`. Use modern Python type annotations (`list[dict]`, `Path | None`), `pathlib.Path`, f-strings, and Google-style docstrings.
- **Geospatial conventions**:
  - `rasterio` transforms map `(col, row) -> (lon, lat)` via `transform * (col, row)`.
  - Ensure `shapely` geometries remain valid after projection/clipping (`make_valid` / `clean_geometry_strict`).
  - Use `billiard.Pool` (not `multiprocessing.Pool`) for parallel post-processing inside Celery workers.

### Frontend (React / TypeScript)
- Functional components with React hooks; shared slicing parameter state via `slicerReducer.ts`.
- Strict TypeScript (`tsc -b` must pass with no `any` or unused locals); lint with ESLint (`npm --prefix frontend run lint`).
- Dispose Three.js geometries/materials in `useEffect` cleanup hooks to avoid WebGL memory leaks.

### Testing
- Mock external network calls (AWS S3 SRTM, SwissTopo STAC, EMODnet, GEBCO, Overpass, Nominatim, CartoDB tiles) and `billiard.Pool` in unit tests.
- Always use `pytest`'s `tmp_path` fixture for temporary files and cache directories.
