from django.conf import settings

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.geocoding import compute_utm_bounds_from_wgs84
from core.utils.slicer import (
    filter_small_features,
    generate_contours,
    mosaic_and_crop,
    project_geometry,
    scale_and_center_contours_to_substrate,
    smooth_geometry,
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
        contours = generate_contours(
            elevation,
            transform,
            self.height,
            self.simplify,
            debug_image_path=settings.DEBUG_IMAGE_PATH,
            center=self.center,
            scale=100,
            bounds=self.bounds,
        )
        # Project, smooth, and scale the contours
        contours = project_geometry(contours, cx, cy, simplify_tolerance=self.simplify)
        contours = smooth_geometry(contours, self.smoothing)
        utm_bounds = compute_utm_bounds_from_wgs84(
            lon_min, lat_min, lon_max, lat_max, cx, cy
        )
        contours = scale_and_center_contours_to_substrate(
            contours, self.substrate_size, utm_bounds
        )
        contours = filter_small_features(contours, self.min_area)
        for contour in contours:
            contour["thickness"] = self.layer_thickness / 1000.0
        return contours
