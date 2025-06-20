"""Compatibility wrapper importing helpers from dedicated modules."""

from .dem import (
    clean_srtm_dem,
    fill_nans_in_dem,
    mosaic_and_crop,
    robust_local_outlier_mask,
    round_affine,
    sample_elevation,
)
from .geometry_ops import (
    clean_geometry,
    clean_geometry_strict,
    clip_contours_to_bbox,
    filter_small_features,
    is_almost_closed,
    project_geometry,
    scale_and_center_contours_to_substrate,
    smooth_geometry,
    walk_bbox_between,
)
from .contour_ops import (
    _create_contourf_levels,
    _extract_level_polygons,
    _prepare_meshgrid,
    _plot_contour_layers,
    _compute_layer_bands,
    generate_contours,
    save_debug_contour_polygon,
)

__all__ = [
    "clean_srtm_dem",
    "fill_nans_in_dem",
    "mosaic_and_crop",
    "robust_local_outlier_mask",
    "round_affine",
    "sample_elevation",
    "clean_geometry",
    "clean_geometry_strict",
    "clip_contours_to_bbox",
    "filter_small_features",
    "is_almost_closed",
    "project_geometry",
    "scale_and_center_contours_to_substrate",
    "smooth_geometry",
    "walk_bbox_between",
    "_create_contourf_levels",
    "_extract_level_polygons",
    "_prepare_meshgrid",
    "_plot_contour_layers",
    "_compute_layer_bands",
    "generate_contours",
    "save_debug_contour_polygon",
]
