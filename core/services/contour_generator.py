import logging

import numpy as np
import shapely
from django.conf import settings
from shapely.geometry import box, shape

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.geocoding import compute_utm_bounds_from_wgs84
from core.utils.slicer import (
    clean_srtm_dem,
    clip_contours_to_bbox,
    fill_nans_in_dem,
    filter_small_features,
    generate_contours,
    mosaic_and_crop,
    project_geometry,
    robust_local_outlier_mask,
    scale_and_center_contours_to_substrate,
    smooth_geometry,
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
        contours = project_geometry(contours, cx, cy, simplify_tolerance=self.simplify)
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
        # Filter small features and set layer thickness
        contours = filter_small_features(
            contours, self.min_area, self.min_feature_width
        )
        # _log_contour_info(contours, "After Filtering Small Features")
        for contour in contours:
            contour["thickness"] = self.layer_thickness / 1000.0
        return contours
