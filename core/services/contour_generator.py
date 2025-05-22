from django.conf import settings

from core.utils.download_clip_elevation_tiles import download_srtm_tiles_for_bounds
from core.utils.geocoding import compute_utm_bounds_from_wgs84
from core.utils.slicer import (
    generate_contours,
    mosaic_and_crop,
    project_geometry,
    scale_and_center_contours_to_substrate,
)


class ContourSlicingJob:
    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        height_per_layer: float,
        num_layers: int,
        simplify: float,
        substrate_size_mm: float,
        layer_thickness_mm: float,
        center: tuple[float, float],
    ):
        self.bounds = bounds
        self.height = height_per_layer
        self.num_layers = num_layers
        self.simplify = simplify
        self.substrate_size = substrate_size_mm
        self.layer_thickness = layer_thickness_mm
        self.center = center

    def run(self) -> list[dict]:
        lon_min, lat_min, lon_max, lat_max = self.bounds
        cx, cy = self.center

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
        contours = project_geometry(contours, cx, cy)
        utm_bounds = compute_utm_bounds_from_wgs84(
            lon_min, lat_min, lon_max, lat_max, cx, cy
        )
        contours = scale_and_center_contours_to_substrate(
            contours, self.substrate_size, utm_bounds
        )
        for contour in contours:
            contour["thickness"] = self.layer_thickness / 1000.0
        return contours
