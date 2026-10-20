import concurrent.futures
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import rasterio
import shapely
from django.conf import settings
from shapely.geometry import (
    box,
    mapping,
    shape,
)

from core.utils.contour_ops import generate_contours
from core.utils.geocoding import compute_utm_bounds_from_wgs84
from core.utils.geometry_ops import project_geometry
from core.utils.osm_features import (
    fetch_buildings,
    fetch_roads,
    fetch_waterways,
    normalize_building_geometry,
    normalize_road_geometry,
    normalize_waterway_geometry,
)

logger = logging.getLogger(__name__)

OCEAN_LEVEL_THRESHOLD = (
    15.0  # Meters. Water bodies above this are considered "Inland" and kept as-is.
)


def touch_cache_file(path: Path) -> None:
    """Update atime and mtime of a cache file."""
    try:
        path.touch(exist_ok=True)
    except Exception:
        pass


def enforce_cache_limit(cache_dir: Path, max_bytes: int) -> None:
    """Enforce size limit on a cache directory by deleting oldest files."""
    try:
        files = [p for p in cache_dir.glob("*.npz") if p.is_file()]
        total_size = sum(p.stat().st_size for p in files)
        if total_size <= max_bytes:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        for p in files:
            if total_size <= max_bytes:
                break
            try:
                sz = p.stat().st_size
                p.unlink(missing_ok=True)
                total_size -= sz
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Error enforcing cache limit on {cache_dir}: {e}")


def _is_unmocked_dem_func(func, expected_name: str) -> bool:
    """Check if DEM function is real and not mocked."""
    if getattr(func, "__name__", "") != expected_name:
        return False
    if "mock" in type(func).__name__.lower():
        return False
    code = getattr(func, "__code__", None)
    if code is None:
        return False
    co_filename = getattr(code, "co_filename", "")
    if not co_filename.endswith("dem.py"):
        return False
    return True


def _log_contour_info(contours, process_step: str = "Contour Generation"):
    """Log information about contours.
    Args:
        contours (list): List of contour features.
    """
    for contour in contours:
        geom = shape(contour["geometry"])
        logger.debug(
            "Contour @ %.1f m, process step %s: geom type = %s, area = %.4f, valid = %s",
            contour["elevation"],
            process_step,
            geom.geom_type,
            geom.area,
            geom.is_valid,
        )


def _geometry_feature_count(geom):
    """Return the count of features for logging."""
    if geom is None or geom.is_empty:
        return 0
    if hasattr(geom, "geoms"):
        return len(geom.geoms)
    return 1


class ContourSlicingJob:
    """Class to handle the slicing of elevation data into contour layers.

    This class is responsible for downloading elevation data, generating contours,
    and preparing the data for slicing into layers.

    Attributes:
        bounds (tuple): Bounding box (lon_min, lat_min, lon_max, lat_max).
        height (float): Target height of the model per layer in mm (determines scale if fixed height).
        num_layers (int): Number of layers to generate.
        simplify (float): Tolerance in meters for simplifying geometry.
        substrate_size (float): Side length of the substrate in mm.
        layer_thickness (float): Thickness of the material in mm.
        center (tuple): Center coordinate (lon, lat).
        smoothing (int): Gaussian smoothing factor for the DEM.
        min_area (float): Minimum area in degrees^2 to keep a polygon.
        min_feature_width (float): Minimum feature width in mm for cleaning.
        fixed_elevation (float | None): Optional fixed elevation for water/lake.
        water_polygon (Polygon | None): Water body geometry if applicable.
        include_roads (bool): Whether to fetch and include roads.
        include_buildings (bool): Whether to fetch and include buildings.
        include_waterways (bool): Whether to fetch and include rivers/streams.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        height_per_layer: float,
        num_layers: int,
        simplify: float,
        substrate_size_mm: float,
        layer_thickness_mm: float,
        center: tuple[float, float],
        smoothing: int,
        min_area: float,
        min_feature_width_mm: float,
        fixed_elevation: float | None = None,
        water_polygon: dict | None = None,
        include_roads: bool = False,
        include_buildings: bool = False,
        include_waterways: bool = False,
        include_bathymetry: bool = False,
        use_osm_water_mask: bool = True,
    ):
        """Initialize the ContourSlicingJob with parameters.
        Args:
            bounds (tuple[float, float, float, float]): Bounding box coordinates (lon_min, lat_min, lon_max, lat_max).
            height_per_layer (float): Height of each layer in mm.
            num_layers (int): Number of layers to generate.
            simplify (float): Simplification tolerance for contours.
            substrate_size_mm (float): Size of the substrate in mm.
            layer_thickness_mm (float): Thickness of each layer in mm.
            center (tuple[float, float]): Center coordinates (lon_center, lat_center).
            smoothing (int): Smoothing factor for contours.
            min_area (float): Minimum area for filtering small features.
            min_feature_width_mm (float): Minimum feature width in mm.
            fixed_elevation (float | None): Need to start slicing from here, if provided.
            water_polygon (dict | None): GeoJSON-like dict representing a water polygon to be used for slicing.
            include_roads (bool): Whether to include road features in the contours.
            include_buildings (bool): Whether to include building features in the contours.
            include_waterways (bool): Whether to include rivers, streams and canals.
        """
        self.bounds = bounds
        self.height_per_layer = height_per_layer
        self.num_layers = num_layers
        self.simplify_tolerance = simplify
        self.substrate_size_mm = substrate_size_mm
        self.layer_thickness_mm = layer_thickness_mm
        self.center = center
        self.smoothing_sigma = smoothing
        self.min_area = min_area
        self.min_feature_width_mm = min_feature_width_mm
        self.include_bathymetry = include_bathymetry
        self.use_osm_water_mask = use_osm_water_mask
        self.fixed_elevation = fixed_elevation
        if water_polygon:
            lon_min, lat_min, lon_max, lat_max = bounds
            epsilon = 0.004
            wgs_bbox = box(
                lon_min - epsilon,
                lat_min - epsilon,
                lon_max + epsilon,
                lat_max + epsilon,
            )
            poly = shape(water_polygon)
            if not poly.is_valid:
                poly = shapely.validation.make_valid(poly)
            cropped = poly.intersection(wgs_bbox)
            self.water_polygon = cropped if not cropped.is_empty else None
        else:
            self.water_polygon = None
        self.include_roads = include_roads
        self.include_buildings = include_buildings
        self.include_waterways = include_waterways

    def _prepare_osm_feature_geom(
        self,
        fetch_fn,
        projection,
        center_x,
        center_y,
        scale_factor,
        cx,
        cy,
        log_label=None,
    ):
        """Fetch and project optional OSM features.

        Args:
            fetch_fn: Callable returning a geometry for the job bounds.
            projection: Existing projection tuple from :func:`project_geometry`.
            center_x: Translation offset in metres (x-axis).
            center_y: Translation offset in metres (y-axis).
            scale_factor: Scaling factor applied after translation.
            cx: Center longitude in WGS84.
            cy: Center latitude in WGS84.
            log_label: Optional label used in log messages.

        Returns:
            A transformed geometry or ``None`` if fetching failed.
        """
        try:
            geom = fetch_fn(self.bounds)
            if log_label:
                count = _geometry_feature_count(geom)
                logger.info(
                    "[OSM] Downloaded %d %s features (%s), total length=%.1f",
                    count,
                    log_label,
                    geom.geom_type,
                    geom.length if hasattr(geom, "length") else -1,
                )
            if not geom.is_empty:
                projected, _ = project_geometry(
                    [{"geometry": mapping(geom), "elevation": 0}],
                    cx,
                    cy,
                    0,
                    projection,
                )
                if projected:
                    g = shape(projected[0]["geometry"])
                    g = shapely.affinity.translate(g, xoff=-center_x, yoff=-center_y)
                    g = shapely.affinity.scale(
                        g, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
                    )
                    return g
            return None
        except Exception as e:
            logger.warning(
                f"Failed to fetch {log_label or 'feature'}: %s", e, exc_info=True
            )
            return None

    def _prepare_osm_road_geoms(
        self,
        projection,
        center_x,
        center_y,
        scale_factor,
        cx,
        cy,
    ) -> dict[str, shapely.geometry.MultiLineString]:
        """Fetch and project road geometries grouped by type."""

        try:
            raw = fetch_roads(self.bounds)
            result = {}
            for rtype, geom in raw.items():
                projected = self._prepare_osm_feature_geom(
                    lambda _b, g=geom: g,
                    projection,
                    center_x,
                    center_y,
                    scale_factor,
                    cx,
                    cy,
                    f"road:{rtype}",
                )
                if projected is not None:
                    result[rtype] = projected
            return result
        except Exception as exc:
            logger.warning("Failed to fetch roads: %s", exc, exc_info=True)
            return {}

    def _fetch_and_prep_dem(self, progress_callback=None):
        """Fetch, merge, and clean DEM data including bathymetry."""
        from core.services.bathymetry_service import BathymetryFetcher
        from core.utils.dem import (
            clean_srtm_dem,
            download_elevation_tiles_for_bounds,
            fill_nans_in_dem,
            get_great_lakes_surface_elevation,
            is_in_great_lakes,
            merge_srtm_and_etopo,
            mosaic_and_crop,
        )
        from core.utils.swiss_topo_api import get_swiss_lake_surface_elevation

        def report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        in_great_lakes = is_in_great_lakes(self.bounds)
        swiss_lake_elev = get_swiss_lake_surface_elevation(self.bounds)
        if in_great_lakes:
            water_surface = float(get_great_lakes_surface_elevation(self.bounds))
        elif swiss_lake_elev is not None:
            water_surface = float(swiss_lake_elev)
        else:
            water_surface = 0.0

        water_threshold = water_surface + OCEAN_LEVEL_THRESHOLD
        skip_ocean_bathy = swiss_lake_elev is not None

        is_real = _is_unmocked_dem_func(
            download_elevation_tiles_for_bounds, "download_elevation_tiles_for_bounds"
        ) and _is_unmocked_dem_func(mosaic_and_crop, "mosaic_and_crop")

        lon_min, lat_min, lon_max, lat_max = self.bounds
        cache_dir = Path(
            getattr(
                settings, "PROCESSED_DEM_CACHE_DIR", Path("data/processed_dem_cache")
            )
        )
        cache_key = (
            hashlib.md5(
                f"{lon_min:.6f}_{lat_min:.6f}_{lon_max:.6f}_{lat_max:.6f}_{int(self.include_bathymetry)}_{int(self.use_osm_water_mask)}".encode()
            ).hexdigest()
            + ".npz"
        )
        cache_file = cache_dir / cache_key

        if is_real and cache_file.exists():
            try:
                touch_cache_file(cache_file)
                with np.load(cache_file) as npz:
                    elevation = npz["elevation"]
                    transform = rasterio.Affine(*npz["transform"])
                masked_elevation = np.ma.masked_invalid(elevation)
                logger.info(f"Loaded prepared DEM from cache: {cache_file}")
                report("Loaded prepared DEM from cache...", 20)
                return elevation, masked_elevation, transform
            except Exception as e:
                logger.warning(
                    f"Failed to load prepared DEM cache {cache_file}: {e}. Removing."
                )
                cache_file.unlink(missing_ok=True)

        provider = "osm"
        bathy_future = None
        osm_future = None

        if not skip_ocean_bathy and (
            self.include_bathymetry or self.use_osm_water_mask
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as io_executor:
                if self.include_bathymetry:

                    def _fetch_bathy():
                        try:
                            fetcher = BathymetryFetcher()
                            return fetcher.fetch_elevation_for_bounds(self.bounds)
                        except Exception as e:
                            logger.error(
                                f"Failed to fetch bathymetry: {e}. Falling back to SRTM only.",
                                exc_info=True,
                            )
                            return None

                    bathy_future = io_executor.submit(_fetch_bathy)

                if self.use_osm_water_mask:
                    import core.utils.osm_tiles as osm_tiles_module

                    osm_future = io_executor.submit(
                        osm_tiles_module.fetch_osm_static_image,
                        self.bounds,
                        zoom=15,
                        provider=provider,
                    )

                report("Downloading elevation tiles...", 5)
                # 1. Fetch Land DEM (SRTM / Swiss)
                try:
                    tile_paths = download_elevation_tiles_for_bounds(
                        self.bounds, include_bathymetry=self.include_bathymetry
                    )
                except TypeError:
                    tile_paths = download_elevation_tiles_for_bounds(self.bounds)

                report("Merging and processing DEM...", 15)
                land_raw, transform = mosaic_and_crop(tile_paths, self.bounds)
        else:
            report("Downloading elevation tiles...", 5)
            # 1. Fetch Land DEM (SRTM / Swiss)
            try:
                tile_paths = download_elevation_tiles_for_bounds(
                    self.bounds, include_bathymetry=self.include_bathymetry
                )
            except TypeError:
                tile_paths = download_elevation_tiles_for_bounds(self.bounds)

            report("Merging and processing DEM...", 15)
            land_raw, transform = mosaic_and_crop(tile_paths, self.bounds)

        # 2. Fetch Bathymetry (if needed for ocean)
        bathy_raw = None
        if self.include_bathymetry and not skip_ocean_bathy:
            report("Fetching global bathymetry (ETOPO)...", 18)
            if bathy_future is not None:
                bathy_raw = bathy_future.result()
            else:
                try:
                    fetcher = BathymetryFetcher()
                    bathy_raw = fetcher.fetch_elevation_for_bounds(self.bounds)
                except Exception as e:
                    logger.error(
                        f"Failed to fetch bathymetry: {e}. Falling back to SRTM only.",
                        exc_info=True,
                    )
                    bathy_raw = None

            if bathy_raw is not None:
                logger.info(
                    f"ETOPO fetched stats: min={bathy_raw.min():.2f}, max={bathy_raw.max():.2f}"
                )

        # 3. Apply Water Masking (OSM Tiles)
        if self.use_osm_water_mask:
            try:
                logger.info("Applying OSM Water Mask from Tiles...")
                from rasterio.warp import Resampling, reproject

                from core.utils.osm_tiles import (
                    TILE_PROVIDERS,
                    fetch_osm_static_image,
                    generate_water_mask_from_tiles,
                )

                # Use CartoDB Voyager No Labels for cleaner mask (no text)
                water_color = TILE_PROVIDERS[provider]["water_color"]

                # Fetch tiles
                # Zoom 15 is good balance of detail and speed
                if osm_future is not None:
                    osm_img, src_transform, src_crs = osm_future.result()
                else:
                    osm_img, src_transform, src_crs = fetch_osm_static_image(
                        self.bounds, zoom=15, provider=provider
                    )

                # Generate mask (True=Water, False=Land)
                # Note: We still apply morphological closing in generate_water_mask_from_tiles
                # which is good for any remaining small artifacts.
                water_mask_src = generate_water_mask_from_tiles(
                    osm_img, water_color=water_color
                )

                # Reproject mask to match land_raw (SRTM)
                # Use uint8 for rasterio compatibility (GDAL Byte)
                water_mask_dest = np.zeros(land_raw.shape, dtype=np.uint8)

                # We need to reproject the boolean mask.
                # Converting to uint8 for reprojection is safer.
                reproject(
                    source=water_mask_src.astype(np.uint8),
                    destination=water_mask_dest,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                )

                # Apply Mask: Where Mask is True (Water), set Land to NaN so it gets filled/merged with Bathy later
                # Ensure land_raw can hold NaNs
                if not np.issubdtype(land_raw.dtype, np.floating):
                    land_raw = land_raw.astype(np.float32)

                # Set water pixels to NaN ONLY if they are likely Ocean (low elevation)
                # Inland rivers/lakes (high elevation) should NOT be masked out.
                # We use the raw SRTM data for this check.

                # Identify where:
                # 1. OSM says Water (mask_dest is True)
                # 2. SRTM says Elevation < Threshold (likely Ocean/Coast)
                # We use a broad threshold (15m) to catch tidal zones and low-lying deltas.
                # High altitude lakes will have elevation >> 15m and thus be skipped.

                # Note: SRTM voids are -32768, which is < 15, so voids in lakes will unfortunately be masked.
                # But ETOPO fill will likely be 0 or negative, which is acceptable for a "void" lake.

                ocean_candidates = water_mask_dest.astype(bool) & (
                    land_raw < water_threshold
                )

                land_raw[ocean_candidates] = np.nan
                logger.info(
                    f"OSM Water Mask applied: {np.count_nonzero(ocean_candidates)} pixels forced to NaN (Elevation < {water_threshold}m)."
                )

            except Exception as e:
                logger.error(f"Failed to apply OSM water mask: {e}", exc_info=True)

        # 4. Merge SRTM and Bathymetry
        if self.include_bathymetry and bathy_raw is not None:
            report("Merging SRTM and High-Res Bathymetry...", 20)
            # If we used the OSM Tile Mask, we pass it as the land mask (Inverse of Water Mask)
            # This ensures merge_srtm_and_etopo uses the EXACT same mask.
            land_mask = None
            if self.use_osm_water_mask and "water_mask_dest" in locals():
                # water_mask_dest is 1 for Water, 0 for Land.
                # merge_srtm_and_etopo expects land_mask where True = Land.
                land_mask = water_mask_dest == 0

            elevation = merge_srtm_and_etopo(
                land_raw, bathy_raw, land_mask=land_mask, transform=transform
            )

            # Enforce Water Mask on Merged Data
            # If we have a water mask, any pixel in it MUST be underwater.
            # ETOPO/Bathymetry might be positive (land) due to resolution issues.
            if self.use_osm_water_mask and "water_mask_dest" in locals():
                mask_bool = water_mask_dest.astype(bool)
                # Identify "Bad Ocean" (Water mask says Yes, Elevation says Land > water_surface - 0.1m)
                # We also check elevation < Threshold to avoid flattening inland rivers.
                bad_ocean = (
                    mask_bool
                    & (elevation > water_surface - 0.1)
                    & (elevation < water_threshold)
                )
                count = np.count_nonzero(bad_ocean)

                if count > 0:
                    clamp_val = water_surface - 5.0
                    elevation[bad_ocean] = clamp_val
                    logger.info(
                        f"Enforced water mask: {count} pixels clamped to {clamp_val}m to prevent 'Vanishing Island' artifacts."
                    )
        else:
            elevation = land_raw

        # 5. Clean the elevation data
        if not np.issubdtype(elevation.dtype, np.floating):
            elevation = elevation.astype(np.float32)

        if self.include_bathymetry:
            min_valid = -11000
        else:
            COAST_OFFSET = water_surface - 1.0
            if self.use_osm_water_mask and "ocean_candidates" in locals():
                elevation[ocean_candidates] = COAST_OFFSET
            elevation[elevation == -32768] = COAST_OFFSET
            elevation[elevation < water_surface] = COAST_OFFSET
            min_valid = COAST_OFFSET

        elevation = clean_srtm_dem(elevation, min_valid=min_valid)

        # Fill any remaining NaN voids in the DEM and write back to elevation
        masked_elevation = np.ma.masked_invalid(elevation)
        filled = fill_nans_in_dem(masked_elevation)
        elevation = (
            filled.filled(np.nan)
            if isinstance(filled, np.ma.MaskedArray)
            else np.asarray(filled)
        )

        logger.debug(
            "Elevation max, min: %.2f, %.2f",
            float(np.nanmax(elevation))
            if not np.isnan(elevation).all()
            else float("nan"),
            float(np.nanmin(elevation))
            if not np.isnan(elevation).all()
            else float("nan"),
        )

        # Cache result if real DEM functions were used (and bathymetry succeeded if requested)
        if (
            is_real
            and isinstance(transform, rasterio.Affine)
            and (
                not self.include_bathymetry or bathy_raw is not None or skip_ocean_bathy
            )
        ):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_file = cache_dir / f"{cache_file.stem}.{os.getpid()}.tmp.npz"
                np.savez_compressed(
                    tmp_file,
                    elevation=elevation,
                    transform=np.array(transform[:6]),
                )
                tmp_file.replace(cache_file)
                enforce_cache_limit(
                    cache_dir,
                    int(getattr(settings, "PROCESSED_DEM_CACHE_MAX_MB", 512))
                    * 1024
                    * 1024,
                )
            except Exception as e:
                logger.warning(f"Failed to cache prepared DEM to {cache_file}: {e}")
                if "tmp_file" in locals() and tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)

        return elevation, masked_elevation, transform

    def _generate_base_contours(
        self, elevation, masked_elevation, transform, progress_callback=None
    ):
        from core.utils.dem import get_great_lakes_surface_elevation, is_in_great_lakes
        from core.utils.swiss_topo_api import get_swiss_lake_surface_elevation

        if progress_callback:
            progress_callback("Generating base contours...", 25)
        effective_fixed_elev = self.fixed_elevation
        if effective_fixed_elev is None:
            if is_in_great_lakes(self.bounds):
                effective_fixed_elev = float(
                    get_great_lakes_surface_elevation(self.bounds)
                )
            else:
                swiss_lake_elev = get_swiss_lake_surface_elevation(self.bounds)
                if swiss_lake_elev is not None:
                    effective_fixed_elev = float(swiss_lake_elev)
        contours = generate_contours(
            masked_elevation_data=masked_elevation,
            elevation_data=elevation,
            transform=transform,
            interval=self.height_per_layer,
            simplify=float(self.simplify_tolerance) / 100000.0,
            debug_image_path=settings.DEBUG_IMAGE_PATH,
            center=self.center,
            bounds=self.bounds,
            fixed_elevation=effective_fixed_elev,
            num_layers=self.num_layers,
            water_polygon=self.water_polygon,
            resolution_scale=1.0,
            min_area_deg2=1e-10,
            dem_smoothing=max(0, float(self.smoothing_sigma) / 5.0),
        )
        _log_contour_info(contours, "After Contour Generation")
        return contours

    def _run_post_processing(
        self, contours, proj_params, utm_bounds, cx, cy, progress_callback=None
    ):
        """Run parallel post-processing on contours (projection, cleanup)."""
        if not contours:
            return []

        def report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        # Prepare arguments for parallel execution
        # Use billiard.Pool because Celery workers are daemonic and cannot spawn children with standard multiprocessing
        # billiard is Celery's fork of multiprocessing that handles this safely.
        import billiard

        from core.utils.parallel_ops import process_and_scale_single_contour

        report(f"Processing {len(contours)} contours in parallel...", 40)
        logger.info(
            f"Starting parallel processing of {len(contours)} contours using billiard..."
        )

        # Prepare arguments for starmap
        tasks_args = [
            (
                c,
                proj_params,
                cx,
                cy,
                self.simplify_tolerance,
                max(0, float(self.smoothing_sigma) / 10.0),
                utm_bounds,
                self.substrate_size_mm,
                self.min_area,
                self.min_feature_width_mm,
            )
            for c in contours
        ]

        processed_contours = []
        try:
            import multiprocessing
            import sys

            cpu_count = multiprocessing.cpu_count()
            workers = max(1, cpu_count - 1)

            chunk_size = max(1, len(tasks_args) // 10)
            chunks = [
                tasks_args[i : i + chunk_size]
                for i in range(0, len(tasks_args), chunk_size)
            ]
            total_chunks = len(chunks)

            if sys.platform == "darwin" and getattr(
                billiard.Pool, "__module__", ""
            ).startswith("billiard"):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:
                    for i, chunk in enumerate(chunks):
                        current_pct = 40 + int(50 * (i / total_chunks))
                        report(
                            f"Slicing contours... (Batch {i + 1}/{total_chunks})",
                            current_pct,
                        )
                        results = list(
                            executor.map(
                                lambda args: process_and_scale_single_contour(*args),
                                chunk,
                            )
                        )
                        processed_contours.extend([r for r in results if r is not None])
            else:
                with billiard.Pool(processes=workers) as pool:
                    for i, chunk in enumerate(chunks):
                        current_pct = 40 + int(50 * (i / total_chunks))
                        report(
                            f"Slicing contours... (Batch {i + 1}/{total_chunks})",
                            current_pct,
                        )
                        results = pool.starmap(process_and_scale_single_contour, chunk)
                        processed_contours.extend([r for r in results if r is not None])

        except Exception as exc:
            logger.error(f"Parallel execution failed: {exc}.", exc_info=True)
            raise exc

        # Sort by elevation
        processed_contours.sort(key=lambda x: x["elevation"])
        return processed_contours

    def _fetch_osm_features(self, utm_bounds, proj_params, cx, cy):
        """Fetch and project OSM features (roads, waterways, buildings)."""
        # We need to re-calculate the scale params to process OSM features identically
        # logic duplicated from `scale_and_center_contours_to_substrate`
        substrate_m = self.substrate_size_mm / 1000.0
        minx, miny, maxx, maxy = utm_bounds
        width = maxx - minx
        height = maxy - miny
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        scale_factor = substrate_m / max(width, height)

        roads_geom = {}
        waterways_geom = None
        buildings_geom = None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            if self.include_roads:
                futures["roads"] = executor.submit(
                    self._prepare_osm_road_geoms,
                    proj_params,
                    center_x,
                    center_y,
                    scale_factor,
                    cx,
                    cy,
                )

            if self.include_waterways:
                futures["waterways"] = executor.submit(
                    self._prepare_osm_feature_geom,
                    fetch_waterways,
                    proj_params,
                    center_x,
                    center_y,
                    scale_factor,
                    cx,
                    cy,
                    "waterway",
                )

            if self.include_buildings:
                futures["buildings"] = executor.submit(
                    self._prepare_osm_feature_geom,
                    fetch_buildings,
                    proj_params,
                    center_x,
                    center_y,
                    scale_factor,
                    cx,
                    cy,
                    "building",
                )

            for key, future in futures.items():
                try:
                    result = future.result()
                    if key == "roads":
                        roads_geom = result
                    elif key == "waterways":
                        waterways_geom = result
                    elif key == "buildings":
                        buildings_geom = result
                except Exception as e:
                    logger.error(f"Failed to fetch {key}: {e}", exc_info=True)

        return roads_geom, waterways_geom, buildings_geom

    def run(self, progress_callback=None) -> list[dict]:
        """Execute the contour slicing job.

        Args:
            progress_callback: Optional function(status_msg, percent) to report progress.

        Returns:
            A list of contour feature dictionaries prepared for slicing.
        """

        def report(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        # Compute projection parameters and UTM bounds before DEM preparation
        lon_min, lat_min, lon_max, lat_max = self.bounds
        cx, cy = self.center

        # Establish projection parameters once on main thread
        _, (proj, rot_center, rot_angle) = project_geometry(
            [], cx, cy, simplify_tolerance=0.0
        )
        utm_bounds = compute_utm_bounds_from_wgs84(
            lon_min, lat_min, lon_max, lat_max, cx, cy
        )
        proj_params = (proj, rot_center, rot_angle)

        osm_executor = None
        osm_future = None
        if self.include_roads or self.include_waterways or self.include_buildings:
            osm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            osm_future = osm_executor.submit(
                self._fetch_osm_features, utm_bounds, proj_params, cx, cy
            )

        try:
            # 1. Fetch and Prepare DEM
            elevation, masked_elevation, transform = self._fetch_and_prep_dem(
                progress_callback
            )

            # 2. Generate Base Contours
            contours = self._generate_base_contours(
                elevation, masked_elevation, transform, progress_callback
            )

            # 3. Post-Processing (Parallel)
            contours = self._run_post_processing(
                contours, proj_params, utm_bounds, cx, cy, progress_callback
            )

            report("Finalizing OSM features...", 90)
            logger.debug(
                f"Parallel processing complete. {len(contours)} valid contours remaining."
            )

            # Step 4: Collect OSM features
            if osm_future is not None:
                roads_geom, waterways_geom, buildings_geom = osm_future.result()
            elif self.include_roads or self.include_waterways or self.include_buildings:
                roads_geom, waterways_geom, buildings_geom = self._fetch_osm_features(
                    utm_bounds, proj_params, cx, cy
                )
            else:
                roads_geom, waterways_geom, buildings_geom = {}, None, None
        finally:
            if osm_executor is not None:
                osm_executor.shutdown(wait=False)

        logger.debug(f"Final Count: {len(contours)} contours")

        n_layers = len(contours)
        if n_layers == 0:
            logger.warning("No contours remaining after processing.")
            return []

        layer_geoms = (
            [shape(c["geometry"]) for c in contours]
            if (self.include_roads or self.include_waterways or self.include_buildings)
            else []
        )

        from core.utils.dem import get_great_lakes_surface_elevation, is_in_great_lakes
        from core.utils.swiss_topo_api import get_swiss_lake_surface_elevation

        swiss_lake_elev = get_swiss_lake_surface_elevation(self.bounds)
        active_water_level = None
        if self.include_bathymetry:
            if is_in_great_lakes(self.bounds):
                active_water_level = float(
                    get_great_lakes_surface_elevation(self.bounds)
                )
            elif swiss_lake_elev is not None:
                active_water_level = float(swiss_lake_elev)

        for idx, contour in enumerate(contours):
            # Recalculate thickness (was done in sequential loop)
            contour["thickness"] = self.layer_thickness_mm / 1000.0
            if active_water_level is not None:
                contour["water_level"] = active_water_level

            # OSM intersections (still sequential, usually fast)
            if self.include_roads or self.include_waterways or self.include_buildings:
                band_i = layer_geoms[idx]
                band_above = layer_geoms[idx + 1] if idx + 1 < n_layers else None
                visible_area = (
                    band_i if band_above is None else band_i.difference(band_above)
                )

                if visible_area.is_empty:
                    if self.include_roads and roads_geom:
                        contour["roads"] = None
                    if self.include_waterways and waterways_geom is not None:
                        contour["waterways"] = None
                    if self.include_buildings and buildings_geom is not None:
                        contour["buildings"] = None
                    continue

                shapely.prepare(visible_area)

                if self.include_roads and roads_geom:
                    road_features = {}
                    for rtype, rgeom in roads_geom.items():
                        clipped = rgeom.intersection(visible_area)
                        normalized = normalize_road_geometry(clipped)
                        if normalized is not None:
                            road_features[rtype] = mapping(normalized)
                    contour["roads"] = road_features if road_features else None
                if self.include_waterways and waterways_geom is not None:
                    clipped = waterways_geom.intersection(visible_area)
                    normalized = normalize_waterway_geometry(clipped)
                    contour["waterways"] = (
                        mapping(normalized) if normalized is not None else None
                    )
                if self.include_buildings and buildings_geom is not None:
                    clipped = buildings_geom.intersection(visible_area)
                    normalized = normalize_building_geometry(clipped)
                    contour["buildings"] = (
                        mapping(normalized) if normalized is not None else None
                    )

        return contours
