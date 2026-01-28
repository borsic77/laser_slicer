import logging

import shapely
import shapely.affinity
from shapely.geometry import LineString, MultiPolygon, box, mapping, shape
from shapely.ops import transform, unary_union

from core.utils.geometry_ops import clean_geometry_strict

logger = logging.getLogger(__name__)


# Standalone function for parallel processing
def process_and_scale_single_contour(
    contour: dict,
    proj_params: tuple,
    center_lon: float,
    center_lat: float,
    simplify_tolerance: float,
    smoothing_radius: int,
    utm_bounds: tuple[float, float, float, float],
    substrate_size_mm: float,
    min_area: float,
    min_feature_width: float,
) -> dict | None:
    """
    Process a single contour through the full geometry pipeline.
    Designed to be picklable for multiprocessing.

    Returns:
        Processed contour dictionary, or None if filtered away/invalid.
    """
    try:
        # 1. Project & Rotate
        # We need to reimplement the core logic of project_geometry here for a single item
        # or use a helper that takes the transformer.
        # Unpacking proj_params: (proj, center_point, rot_angle)
        # Note: 'proj' from pyproj might not be picklable depending on version?
        # Actually pyproj.Transformer IS picklable in recent versions.
        # But 'center_point' is a shapely Point which IS picklable.

        transformer, rot_center, rot_angle = proj_params

        geom = shape(contour["geometry"])
        # Transform (WGS84 -> UTM)
        projected_geom = transform(transformer.transform, geom)

        # Rotate (North alignment)
        # Standardize: rot_center is the projection of the region center
        rotated_geom = shapely.affinity.rotate(
            projected_geom, -rot_angle, origin=rot_center
        )

        # Clean
        if rotated_geom.geom_type in ("Polygon", "MultiPolygon"):
            cleaned_geom = clean_geometry_strict(rotated_geom)
        elif rotated_geom.geom_type in ("LineString", "MultiLineString"):
            cleaned_geom = rotated_geom if not rotated_geom.is_empty else None
        else:
            # Handle mixed collections if any
            if hasattr(rotated_geom, "geoms"):
                allowed = [
                    g
                    for g in rotated_geom.geoms
                    if g.geom_type
                    in ("Polygon", "MultiPolygon", "LineString", "MultiLineString")
                ]
                cleaned_geom = unary_union(allowed) if allowed else None
            else:
                cleaned_geom = None

        if cleaned_geom is None:
            return None

        # Simplify
        if simplify_tolerance > 0.0:
            cleaned_geom = cleaned_geom.simplify(
                simplify_tolerance, preserve_topology=True
            )

        # 2. Smooth
        if smoothing_radius > 0:
            cleaned_geom = cleaned_geom.buffer(smoothing_radius).buffer(
                -smoothing_radius
            )

        # 3. Clip to UTM bounds
        bbox_poly = box(*utm_bounds)
        clipped_geom = cleaned_geom.intersection(bbox_poly)
        cleaned_geom = clean_geometry_strict(clipped_geom)

        if cleaned_geom is None or cleaned_geom.is_empty:
            return None

        # 4. Scale & Center to Substrate
        substrate_m = substrate_size_mm / 1000.0
        minx, miny, maxx, maxy = utm_bounds
        width = maxx - minx
        height = maxy - miny
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        scale_factor = substrate_m / max(width, height)

        # Translate to origin relative to bounding box center
        moved = shapely.affinity.translate(cleaned_geom, xoff=-center_x, yoff=-center_y)
        # Scale
        scaled = shapely.affinity.scale(
            moved, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
        )

        # 5. Filter Small Features
        min_area_m2 = (min_area / 1e4) * (
            scale_factor**2
        )  # min_area is usually cm^2? Wait.
        # In filter_small_features input min_area is "cm2".
        # But we are now in "scaled substrate space" (meters?).
        # Actually logic in `scale_and_center` puts it in meters relative to substrate center?
        # WAIT. logic in `filter_small_features` uses `min_area_cm2 / 1e4` -> m^2.
        # But the geometry `scaled` is in... what units?
        # substrate_size_mm / 1000.0 -> meters.
        # So yes, `scaled` is in Real World Meters of the PHYSICAL OBJECT.
        # So 1 unit = 1 meter of plywood.
        # min_area_cm2 / 1e4 gives m^2. Correct.

        target_min_area_m2 = min_area / 1e4 if min_area > 0 else 0.0

        # Min width is also in mm, so convert to m
        target_min_width_m = (
            min_feature_width / 1000.0 if min_feature_width > 0 else 0.0
        )

        final_geom = scaled

        if target_min_width_m > 0:
            try:
                final_geom = final_geom.buffer(-target_min_width_m / 2).buffer(
                    target_min_width_m / 2
                )
                final_geom = clean_geometry_strict(final_geom)
                if final_geom is None or final_geom.is_empty:
                    return None
            except Exception:
                pass

        # Filter by area
        parts = []
        if final_geom.geom_type == "Polygon":
            if final_geom.area >= target_min_area_m2:
                parts.append(final_geom)
        elif final_geom.geom_type == "MultiPolygon":
            parts.extend(g for g in final_geom.geoms if g.area >= target_min_area_m2)

        if not parts:
            return None

        final_geom = MultiPolygon(parts)

        # Prepare result
        result = contour.copy()
        result["geometry"] = mapping(final_geom)
        result["closed"] = True
        return result

    except Exception as e:
        # logging might not work well in workers depending on setup, but print works
        # logger.error(f"Error processing contour {contour.get('elevation')}: {e}")
        return None
