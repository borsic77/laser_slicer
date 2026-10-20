# Changelog

## v3.1.0 – SwissBATHY3D, Great Lakes Bathymetry, Faster Slicing Pipeline & Bounded Caches

**Laser Slicer v3.1** adds high-resolution **SwissBATHY3D (2m)** lake bathymetry across Switzerland, **Laurentian Great Lakes bathymetry (~90m)**, global ocean bathymetry via NOAA NCEI's ArcGIS ImageServer, a live UI bathymetry dataset indicator, accelerated contour slicing, and bounded LRU disk caches for production deployments.

### SwissBATHY3D & Great Lakes Bathymetry
- **SwissBATHY3D (2m)**: Integration of swisstopo's `ch.swisstopo.swissbathy3d` collection covering all 22 mapped Swiss lakes (Lake Geneva, Lake Constance, Lake Zurich, Lake Lucerne, Lake Thun, Lake Brienz, Lake Neuchatel, Lake Biel, Rotsee, etc.).
- **HTTP Range ZIP Streaming**: `HttpRangeFile` wrapper allows on-demand partial reads of remote multi-hundred megabyte lake archives from `data.geo.admin.ch`, downloading and reprojecting only intersecting 1km x 1km ESRI ASCII Grid tiles without pulling entire archives.
- **NOAA Great Lakes Bathymetry (~90m / 3-arc-second)**: Automatic detection for Lakes Superior, Michigan, Huron, St. Clair, Erie, and Ontario via NOAA NCEI's `DEM_global_mosaic` ImageServer.
- **Surface Datum Fusion & Automatic Shoreline Contours**: Converts lake depths to true elevation above sea level for both Swiss lakes (LN02/LHN95 datum) and Great Lakes (IGLD 1985 datum), inserting exact shoreline contour levels without artificial cliffs.
- **Live UI Dataset Indicator**: Real-time sidebar guidance displays the active bathymetry dataset for the current bounding box (`SwissBATHY3D 2m`, `NOAA Great Lakes 90m`, `EMODnet Europe 115m`, `NOAA Global 450m`, or instructions to use Fix Elevation for other inland water bodies).
- **3D Preview Water Level Support**: Underwater lake layers (`elevation < water_level`) render in the blue bathymetry palette with the translucent 3D water plane positioned at the lake surface.
- **Restored Global Ocean Fallback**: Replaced deprecated static GEBCO/ETOPO file URLs with NOAA NCEI `DEM_global_mosaic` (450m) and `ETOPO1_bedrock` ImageServer GeoTIFF exports alongside EMODnet WCS (115m) for Europe.
- **OSM Water Mask Fix**: Switched water-mask tile fetching from CartoDB's retired unauthenticated endpoint to OpenStreetMap tiles (`#AAD3DF`).

### Slicing Pipeline Performance
- **Prepared-DEM `.npz` Cache**: Caches merged, water-masked, and NaN-filled `(elevation, transform)` rasters so re-slicing the same region with new layer counts, heights, or smoothing takes `< 0.5s`.
- **Parallel & Disk-Cached Water-Mask Tiles**: Caches individual OSM water-mask PNG tiles on disk, fetches uncached tiles across 8 parallel worker threads, and adaptively steps down zoom on large bounding boxes (>36 tiles).
- **Concurrent I/O Overlapping**: Overlaps SRTM tile loading, bathymetry fetching, water-mask tile fetching, and Overpass feature queries concurrently.
- **Vectorized Geometry Hot-Loops**: Vectorized NumPy affine coordinate math in `_contours_to_polygons`, cached `pyproj.Transformer` instances, and `shapely.prepare` acceleration for OSM road/building/waterway clipping.

### Bounded LRU Disk Caches
- **Automatic Cache Eviction**: Enforces configurable size limits with an 85% hysteresis watermark across all 5 cache directories (~3.5 GB total by default):
  - `TILE_CACHE_MAX_MB=1024` (SRTM)
  - `ALTI3D_CACHE_MAX_MB=1024` (SwissALTI3D)
  - `BATHY_CACHE_MAX_MB=512` (Bathymetry)
  - `OSM_CACHE_MAX_MB=512` (Overpass JSON + Water-Mask PNG tiles)
  - `PROCESSED_DEM_CACHE_MAX_MB=512` (Prepared DEM `.npz`)
- Integrated into both on-write cache hooks and the hourly `cleanup_old_media` Celery task / `cleanupmedia` management command.

### Security, Stability & Fixes
- Restored `fixed_elevation` and `water_polygon` lake-cutout support in `generate_contours`.
- Hardened production `SECRET_KEY`, `DEBUG`, and `ALLOWED_HOSTS` settings, export filename sanitization, and input coordinate/bounds validation.
- Fixed Three.js `ExtrudeGeometry` WebGL memory leaks and polling race conditions in the React frontend.
- Fixed macOS (`darwin`) fork-safety crashes in Celery and parallel contour post-processing.

---

## v3.0.0 – Global Bathymetry (Oceans)

### Highlights
- **Underwater Terrain**: Slice ocean floors down to -11,000m using EMODnet / GEBCO / ETOPO 2022 data.
- **Coastline Fusion**: Zero-crossing and OSM tile water-masking pipeline blends land (SRTM) and sea (Bathymetry) for continuous coastal models.
- **Dynamic Range**: Elevation service detects and adapts to negative elevation ranges when bathymetry is enabled.

---

## v2.1.0 – Parallel Processing, UI Improvements & Documentation

### Highlights
- Parallel contour post-processing in Celery workers.
- Interactive 3D preview and UI improvements.
