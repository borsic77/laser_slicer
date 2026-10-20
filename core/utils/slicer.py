"""Compatibility wrapper importing helpers from dedicated modules."""

from .contour_ops import (
    _compute_layer_bands,
    _create_levels,
    generate_contours,
    save_debug_contour_polygon,
)
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

__all__ = [
    "_compute_layer_bands",
    "_create_levels",
    "clean_geometry",
    "clean_geometry_strict",
    "clean_srtm_dem",
    "clip_contours_to_bbox",
    "fill_nans_in_dem",
    "filter_small_features",
    "generate_contours",
    "is_almost_closed",
    "mosaic_and_crop",
    "project_geometry",
    "robust_local_outlier_mask",
    "round_affine",
    "sample_elevation",
    "save_debug_contour_polygon",
    "scale_and_center_contours_to_substrate",
    "smooth_geometry",
    "walk_bbox_between",
]
