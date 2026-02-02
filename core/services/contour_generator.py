import logging

import numpy as np
import shapely
from django.conf import settings
from shapely.geometry import (
    box,
    mapping,
    shape,
)

from core.services.bathymetry_service import BathymetryFetcher
from core.utils.contour_ops import generate_contours
from core.utils.dem import (
    clean_srtm_dem,
    fill_nans_in_dem,
    merge_srtm_and_etopo,
    mosaic_and_crop,
)
from core.utils.download_clip_elevation_tiles import (
    download_elevation_tiles_for_bounds,
)
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
        self.height = height_per_layer
        self.num_layers = num_layers
        self.simplify = simplify
        self.substrate_size = substrate_size_mm
        self.layer_thickness = layer_thickness_mm
        self.center = center
        self.smoothing = smoothing
        self.min_area = min_area
        self.min_feature_width = min_feature_width_mm
        self.fixed_elevation = fixed_elevation
        self.water_polygon = shape(water_polygon) if water_polygon else None
        if water_polygon:
            # Intersect with area of interest in lon/lat before anything else

            lon_min, lat_min, lon_max, lat_max = bounds
            epsilon = 0.004  # Small epsilon to avoid precision issues
            wgs_bbox = box(
                lon_min - epsilon,
                lat_min - epsilon,
                lon_max + epsilon,
                lat_max + epsilon,
            )
            poly = shape(water_polygon)
            cropped = poly.intersection(wgs_bbox)
            self.water_polygon = cropped if not cropped.is_empty else None
        else:
            self.water_polygon = None
        self.include_roads = include_roads
        self.include_buildings = include_buildings
        self.include_waterways = include_waterways
        self.include_bathymetry = include_bathymetry

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

        report("Downloading elevation tiles...", 5)
        # Unpack the bounding box coordinates and center
        lon_min, lat_min, lon_max, lat_max = self.bounds
        cx, cy = self.center
        # Download elevation tiles and generate contours
        tile_paths = download_elevation_tiles_for_bounds(self.bounds)

        report("Merging and processing DEM...", 15)
        elevation, transform = mosaic_and_crop(tile_paths, self.bounds)

        if self.include_bathymetry:
            report("Fetching global bathymetry (ETOPO)...", 18)
            try:
                fetcher = BathymetryFetcher()
                etopo_data = fetcher.fetch_elevation_for_bounds(self.bounds)
                logger.info(
                    f"ETOPO fetched stats: min={etopo_data.min():.2f}, max={etopo_data.max():.2f}"
                )

                report("Merging SRTM and High-Res Bathymetry...", 20)
                elevation = merge_srtm_and_etopo(
                    elevation, etopo_data, bounds=self.bounds, transform=transform
                )
            except Exception as e:
                logger.error(
                    f"Failed to fetch/merge bathymetry: {e}. Falling back to SRTM only.",
                    exc_info=True,
                )
                # Continue with SRTM only

        # Clean the elevation data
        # If bathymetry is disabled, we clamp the ocean/voids to 0 to provide a flat base.
        if self.include_bathymetry:
            min_valid = -11000
        else:
            # Enforce flat ocean: Treat SRTM voids (-32768) and negative noise as a fixed negative value.
            # We use -1.0 (instead of 0) to ensure the coastline (at 0m) is generated as a distinct cut.
            # This creates a "base layer" for the water at -1.0m, and the land starts at 0.0m.
            COAST_OFFSET = -1.0
            elevation[elevation == -32768] = COAST_OFFSET
            elevation[elevation < 0] = COAST_OFFSET
            min_valid = COAST_OFFSET

        elevation = clean_srtm_dem(elevation, min_valid=min_valid)
        # elevation = robust_local_outlier_mask(elevation)
        logger.debug("Elevation max, min: %.2f, %.2f", elevation.max(), elevation.min())

        # Create mask for filling NaNs (clean_srtm_dem produces NaNs for invalid values)
        masked_elevation = np.ma.masked_invalid(elevation)
        masked_elevation = fill_nans_in_dem(masked_elevation)

        report("Generating base contours...", 25)
        contours = generate_contours(
            masked_elevation_data=masked_elevation,  # The filled/clean version
            elevation_data=elevation,  # The raw version (though new impl uses this for thresholding, so maybe we want the clean one?)
            # Actually, looking at new impl:
            # `data_filled = elevation_data.copy(); data_filled[np.isnan] = -inf`
            # So passing raw `elevation` as `elevation_data` is fine, it handles NaNs internally.
            transform=transform,
            interval=self.height,
            simplify=float(self.simplify) / 100000.0,
            debug_image_path=settings.DEBUG_IMAGE_PATH,
            center=self.center,
            # scale removed
            bounds=self.bounds,
            fixed_elevation=self.fixed_elevation,
            num_layers=self.num_layers,
            water_polygon=self.water_polygon,
            resolution_scale=1.0,
            min_area_deg2=1e-10,
            dem_smoothing=max(0, float(self.smoothing) / 5.0),
        )
        _log_contour_info(contours, "After Contour Generation")

        # Establish projection parameters once on main thread
        # We use the center point to define a stable origin and rotation (Grid Convergence)
        # Calling project_geometry with [] will initialize it from cx, cy
        _, (proj, rot_center, rot_angle) = project_geometry(
            [], cx, cy, simplify_tolerance=0.0
        )

        # Prepare arguments for parallel execution
        # Use billiard.Pool because Celery workers are daemonic and cannot spawn children with standard multiprocessing
        # billiard is Celery's fork of multiprocessing that handles this safely.
        import billiard

        from core.utils.parallel_ops import process_and_scale_single_contour

        utm_bounds = compute_utm_bounds_from_wgs84(
            lon_min, lat_min, lon_max, lat_max, cx, cy
        )

        proj_params = (proj, rot_center, rot_angle)

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
                self.simplify,
                # Scale smoothing for geometric buffer: slider (0-200) -> radius (0-20m)
                # Raw value was too aggressive (50m buffer for value 50).
                max(0, float(self.smoothing) / 10.0),
                utm_bounds,
                self.substrate_size,
                self.min_area,
                self.min_feature_width,
            )
            for c in contours
        ]

        processed_contours = []
        try:
            # billiard handles daemon processes properly
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            workers = max(1, cpu_count - 1)

            # Simple manual chunking to report progress during parallel map
            # We can't use starmap if we want granular progress for each item easily without a callback proxy
            # But we can chunk it.

            chunk_size = max(1, len(tasks_args) // 10)
            chunks = [
                tasks_args[i : i + chunk_size]
                for i in range(0, len(tasks_args), chunk_size)
            ]

            total_chunks = len(chunks)

            with billiard.Pool(processes=workers) as pool:
                for i, chunk in enumerate(chunks):
                    # report progress based on chunks
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
        contours = processed_contours

        report("Finalizing OSM features...", 90)
        logger.debug(
            f"Parallel processing complete. {len(contours)} valid contours remaining."
        )

        # Prepare optional OSM features (Roads/Waterways need the 'scale_factor' and 'center' from the now-implicit scaling)
        # We need to re-calculate the scale params to process OSM features identically
        # logic duplicated from `scale_and_center_contours_to_substrate`
        substrate_m = self.substrate_size / 1000.0
        minx, miny, maxx, maxy = utm_bounds
        width = maxx - minx
        height = maxy - miny
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        scale_factor = substrate_m / max(width, height)

        # ... (rest of OSM processing remains sequential as it's usually fast/small) ...

        roads_geom = (
            self._prepare_osm_road_geoms(
                (proj, rot_center, rot_angle),  # projection
                center_x,
                center_y,
                scale_factor,
                cx,
                cy,
            )
            if self.include_roads
            else {}
        )
        waterways_geom = (
            self._prepare_osm_feature_geom(
                fetch_waterways,
                (proj, rot_center, rot_angle),
                center_x,
                center_y,
                scale_factor,
                cx,
                cy,
                "waterway",
            )
            if self.include_waterways
            else None
        )
        buildings_geom = (
            self._prepare_osm_feature_geom(
                fetch_buildings,
                (proj, rot_center, rot_angle),
                center_x,
                center_y,
                scale_factor,
                cx,
                cy,
                "building",
            )
            if self.include_buildings
            else None
        )

        # Filter small features was already done in parallel step!
        # Removed filter_small_features call.

        logger.debug(f"Final Count: {len(contours)} contours")

        n_layers = len(contours)
        if n_layers == 0:
            logger.warning("No contours remaining after processing.")
            return []

        for idx, contour in enumerate(contours):
            # Recalculate thickness (was done in sequential loop)
            contour["thickness"] = self.layer_thickness / 1000.0

            # OSM intersections (still sequential, usually fast)
            if self.include_roads or self.include_waterways or self.include_buildings:
                band_i = shape(contour["geometry"])
                # We need simple visibility check: what is NOT covered by the layer above?
                # This is "slicing logic".
                # Actually previously this was:
                # band_above = ... if idx + 1 < n_layers else None
                # visible = band_i.diff(band_above)
                # ...

                band_above = (
                    shape(contours[idx + 1]["geometry"]) if idx + 1 < n_layers else None
                )
                visible_area = (
                    band_i if band_above is None else band_i.difference(band_above)
                )

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
