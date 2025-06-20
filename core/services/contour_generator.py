import logging

import numpy as np
import shapely
from django.conf import settings
from shapely.geometry import (
    box,
    mapping,
    shape,
)

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.geocoding import compute_utm_bounds_from_wgs84
from core.utils.osm_features import (
    fetch_buildings,
    fetch_roads,
    fetch_waterways,
    normalize_building_geometry,
    normalize_road_geometry,
    normalize_waterway_geometry,
)
from core.utils.dem import clean_srtm_dem, fill_nans_in_dem, mosaic_and_crop
from core.utils.geometry_ops import (
    clip_contours_to_bbox,
    filter_small_features,
    project_geometry,
    scale_and_center_contours_to_substrate,
    smooth_geometry,
)
from core.utils.contour_ops import generate_contours

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
        """ "
        Prepare OSM feature geometry by fetching and transforming it.
        Args:
            fetch_fn (callable): Function to fetch the OSM feature geometry.
            projection (str): Projection string for the geometry.
            center_x (float): Center x-coordinate for translation.
            center_y (float): Center y-coordinate for translation.
            scale_factor (float): Scale factor for the geometry.
            cx (float): Center x-coordinate in WGS84.
            cy (float): Center y-coordinate in WGS84.
            log_label (str | None): Label for logging the feature type.
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

    def run(self) -> list[dict]:
        """Run the contour slicing job.
        This method downloads elevation data, generates contours,
        and prepares the data for slicing into layers.
        Returns:
            list[dict]: List of contour features with their properties.
        """
        # Unpack the bounding box coordinates and center
        lon_min, lat_min, lon_max, lat_max = self.bounds
        cx, cy = self.center
        # Download elevation tiles and generate contours
        tile_paths = download_srtm_tiles_for_bounds(self.bounds)
        elevation, transform = mosaic_and_crop(tile_paths, self.bounds)
        # Clean the elevation data
        elevation = clean_srtm_dem(elevation)
        # elevation = robust_local_outlier_mask(elevation)
        logger.debug("Elevation max, min: %.2f, %.2f", elevation.max(), elevation.min())
        masked_elevation = np.ma.masked_where(
            ~np.isfinite(elevation) | (elevation <= -32768), elevation
        )
        masked_elevation = fill_nans_in_dem(masked_elevation)

        contours = generate_contours(
            elevation,
            masked_elevation,
            transform,
            self.height,
            self.simplify,
            debug_image_path=settings.DEBUG_IMAGE_PATH,
            center=self.center,
            scale=100,
            bounds=self.bounds,
            fixed_elevation=self.fixed_elevation,
            num_layers=self.num_layers,
            water_polygon=self.water_polygon,
        )
        _log_contour_info(contours, "After Contour Generation")
        # Project, smooth, and scale the contours
        contours, projection = project_geometry(
            contours, cx, cy, simplify_tolerance=self.simplify
        )
        _log_contour_info(contours, "After Projection")
        contours = smooth_geometry(contours, self.smoothing)
        _log_contour_info(contours, "After Smoothing")
        utm_bounds = compute_utm_bounds_from_wgs84(
            lon_min, lat_min, lon_max, lat_max, cx, cy
        )
        # clip to make sure an fixed elevation water body does not violate the bounding box
        contours = clip_contours_to_bbox(contours, utm_bounds)
        _log_contour_info(contours, "After Clipping to Bounding Box")

        contours = scale_and_center_contours_to_substrate(
            contours, self.substrate_size, utm_bounds
        )

        # Log contour information
        _log_contour_info(contours, "After Scaling and Centering")
        # Prepare optional OSM features
        substrate_m = self.substrate_size / 1000.0
        minx, miny, maxx, maxy = utm_bounds
        width = maxx - minx
        height = maxy - miny
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        scale_factor = substrate_m / max(width, height)
        roads_geom = (
            self._prepare_osm_feature_geom(
                fetch_roads,
                projection,
                center_x,
                center_y,
                scale_factor,
                cx,
                cy,
                "road",
            )
            if self.include_roads
            else None
        )
        waterways_geom = (
            self._prepare_osm_feature_geom(
                fetch_waterways,
                projection,
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
                projection,
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
        # Filter small features and set layer thickness
        contours = filter_small_features(
            contours, self.min_area, self.min_feature_width
        )
        # _log_contour_info(contours, "After Filtering Small Features")
        n_layers = len(contours)
        for idx, contour in enumerate(contours):
            band_i = shape(contour["geometry"])
            band_above = (
                shape(contours[idx + 1]["geometry"]) if idx + 1 < n_layers else None
            )
            visible_area = (
                band_i if band_above is None else band_i.difference(band_above)
            )

            if self.include_roads and roads_geom is not None:
                clipped = roads_geom.intersection(visible_area)
                normalized = normalize_road_geometry(clipped)
                contour["roads"] = (
                    mapping(normalized) if normalized is not None else None
                )
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

            contour["thickness"] = self.layer_thickness / 1000.0

        return contours
