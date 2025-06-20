"""Geometry helper functions used in contour generation."""

import logging
import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import shapely
import pyproj
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Polygon,
    box,
    mapping,
    shape,
)
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import transform, unary_union
from shapely.validation import make_valid

logger = logging.getLogger(__name__)


def clean_geometry(geom: BaseGeometry) -> BaseGeometry | None:
    """Attempt to fix invalid geometries.

    Args:
        geom: Geometry to clean.

    Returns:
        The cleaned geometry or ``None`` if it cannot be fixed.
    """

    geom = make_valid(geom)
    if geom.is_empty or geom.area == 0:
        return None
    return orient(geom)


def clean_geometry_strict(geom: BaseGeometry) -> BaseGeometry | None:
    """Aggressively fix invalid geometries.

    Args:
        geom: Geometry to validate.

    Returns:
        A valid geometry or ``None`` if fixing fails.
    """
    geom = make_valid(geom)
    if geom.is_empty or geom.area == 0:
        return None
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None
    if geom.is_empty or geom.area == 0:
        return None
    if not geom.is_valid:
        return None
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return None
        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return orient(geom)


def walk_bbox_between(
    coords, start_idx: int, end_idx: int, direction: str = "cw"
) -> list:
    """Walk bbox coordinates between two indices.

    Args:
        coords: Sequence of corner coordinates in clockwise order.
        start_idx: Start index within ``coords``.
        end_idx: End index within ``coords``.
        direction: ``"cw"`` for clockwise, ``"ccw"`` for counter-clockwise.

    Returns:
        List of coordinates along the bounding box path.
    """
    if direction == "cw":
        if start_idx >= end_idx:
            return coords[end_idx : start_idx + 1]
        return coords[end_idx:] + coords[: start_idx + 1]
    if start_idx <= end_idx:
        return coords[start_idx : end_idx + 1][::-1]
    return (coords[start_idx:] + coords[: end_idx + 1])[::-1]


def is_almost_closed(line: LineString, tolerance: float = 1e-8) -> bool:
    """Check if a line is nearly closed.

    Args:
        line: Line to check.
        tolerance: Maximum gap length considered closed.

    Returns:
        ``True`` if the line's endpoints are within ``tolerance`` of each other.
    """
    return (
        line.coords[0] != line.coords[-1]
        and LineString([line.coords[0], line.coords[-1]]).length < tolerance
    )


def _flatten_polygons(geoms: List[Polygon]) -> List[Polygon]:
    """Extract polygon parts from heterogeneous geometries.

    Args:
        geoms: Iterable of polygons or multipolygons.

    Returns:
        A list containing only ``Polygon`` objects.
    """
    flat: List[Polygon] = []
    for geom in geoms:
        if geom.geom_type == "Polygon":
            flat.append(geom)
        elif geom.geom_type == "MultiPolygon":
            flat.extend(g for g in geom.geoms if g.geom_type == "Polygon")
    return flat


def _force_multipolygon(geom):
    """Return a geometry as a ``MultiPolygon``.

    Args:
        geom: Source geometry which may be a ``Polygon`` or ``GeometryCollection``.

    Returns:
        A ``MultiPolygon`` instance (possibly empty).
    """
    if isinstance(geom, (Polygon, MultiPolygon)):
        return MultiPolygon([geom]) if isinstance(geom, Polygon) else geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        return MultiPolygon(polys)
    return MultiPolygon()


def _grid_convergence_angle_from_geometry(projected_geoms: list) -> float:
    """Compute the rotation angle for a set of projected geometries.

    Args:
        projected_geoms: Geometries already in a projected CRS.

    Returns:
        Rotation angle in degrees for north alignment.
    """
    if not projected_geoms:
        return 0.0
    unioned = unary_union(projected_geoms)
    if unioned.is_empty:
        return 0.0
    min_rect = unioned.minimum_rotated_rectangle
    coords = list(min_rect.exterior.coords)
    max_len = 0.0
    angle_deg = 0.0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.hypot(dx, dy)
        if length > max_len:
            max_len = length
            angle_deg = math.degrees(math.atan2(dy, dx))
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg < -180:
        angle_deg += 360
    if abs(angle_deg) > 45:
        angle_deg -= 90 * np.sign(angle_deg)
    return angle_deg


def project_geometry(
    contours: List[dict],
    center_lon: float,
    center_lat: float,
    simplify_tolerance: float = 0.0,
    existing_transform=None,
):
    """Project contour geometries to UTM and rotate north-up.

    Args:
        contours: Contours with GeoJSON geometries in WGS84.
        center_lon: Longitude used to select the UTM zone.
        center_lat: Latitude used to select the UTM zone.
        simplify_tolerance: Optional simplification tolerance in metres.
        existing_transform: Optional precomputed transform triple.

    Returns:
        Tuple of the projected contours and a transformation descriptor.
    """
    if existing_transform is not None:
        proj, center, rot_angle = existing_transform
    else:
        zone_number = int((center_lon + 180) / 6) + 1
        is_northern = center_lat >= 0
        epsg_code = f"326{zone_number:02d}" if is_northern else f"327{zone_number:02d}"
        proj = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg_code}", always_xy=True
        )
        projected_geoms = []
        for contour in contours:
            try:
                geom = shape(contour["geometry"])
                projected_geoms.append(transform(proj.transform, geom))
            except Exception:
                continue
        if not projected_geoms:
            return [], (proj, None, 0.0)
        merged = unary_union(projected_geoms)
        center = merged.centroid
        rot_angle = _grid_convergence_angle_from_geometry(projected_geoms)

    projected_contours = []
    for contour in contours:
        try:
            geom = shape(contour["geometry"])
            projected_geom = transform(proj.transform, geom)
            rotated_geom = shapely.affinity.rotate(
                projected_geom, -rot_angle, origin=center
            )
            if rotated_geom.geom_type in ("Polygon", "MultiPolygon"):
                cleaned_geom = clean_geometry_strict(rotated_geom)
            elif rotated_geom.geom_type in ("LineString", "MultiLineString"):
                cleaned_geom = rotated_geom if not rotated_geom.is_empty else None
            else:
                allowed = [
                    g
                    for g in rotated_geom.geoms
                    if g.geom_type
                    in ("Polygon", "MultiPolygon", "LineString", "MultiLineString")
                ]
                cleaned_geom = unary_union(allowed) if allowed else None
            if cleaned_geom is None:
                continue
            final_geom = cleaned_geom
            if simplify_tolerance > 0.0:
                final_geom = final_geom.simplify(
                    simplify_tolerance, preserve_topology=True
                )
            contour_copy = contour.copy()
            contour_copy["geometry"] = mapping(final_geom)
            projected_contours.append(contour_copy)
        except Exception as exc:  # pragma: no cover - log and skip
            logger.warning(
                "Failed to project contour at elevation %s: %s",
                contour.get("elevation", "unknown"),
                exc,
            )
            continue
    return projected_contours, (proj, center, rot_angle)


def clip_contours_to_bbox(
    contours: list[dict], bbox: Tuple[float, float, float, float]
) -> list[dict]:
    """Clip projected contours to a bounding box.

    Args:
        contours: Contours in projected coordinates.
        bbox: Target bounds ``(minx, miny, maxx, maxy)``.

    Returns:
        List of contours intersecting the box.
    """
    bbox_poly = box(*bbox)
    clipped = []
    for contour in contours:
        geom = shape(contour["geometry"])
        clipped_geom = geom.intersection(bbox_poly)
        cleaned = clean_geometry_strict(clipped_geom)
        if cleaned is not None and not cleaned.is_empty:
            contour_clipped = contour.copy()
            contour_clipped["geometry"] = mapping(cleaned)
            clipped.append(contour_clipped)
    return clipped


def scale_and_center_contours_to_substrate(
    contours: List[dict],
    substrate_size_mm: float,
    utm_bounds: Tuple[float, float, float, float],
) -> List[dict]:
    """Scale and center contours to the print substrate.

    Args:
        contours: Contours in projected coordinates.
        substrate_size_mm: Size of the square substrate in millimetres.
        utm_bounds: Bounding box of the source data in the same projection.

    Returns:
        Updated contour list scaled and centered to the substrate.
    """
    substrate_m = substrate_size_mm / 1000.0
    minx, miny, maxx, maxy = utm_bounds
    width = maxx - minx
    height = maxy - miny
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    scale_factor = substrate_m / max(width, height)
    elevation_groups = defaultdict(list)
    for contour in contours:
        elevation_groups[contour["elevation"]].append(shape(contour["geometry"]))

    updated = []
    for elevation, group_geoms in elevation_groups.items():
        flat_geoms = []
        for g in group_geoms:
            if g.geom_type == "Polygon":
                flat_geoms.append(g)
            elif g.geom_type == "MultiPolygon":
                flat_geoms.extend(g.geoms)
        union_geom = MultiPolygon(flat_geoms)
        moved = shapely.affinity.translate(union_geom, xoff=-center_x, yoff=-center_y)
        scaled = shapely.affinity.scale(
            moved, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
        )
        updated.append(
            {"elevation": elevation, "geometry": mapping(scaled), "closed": True}
        )

    all_scaled_bounds = [shape(c["geometry"]).bounds for c in updated]
    min_x = min(b[0] for b in all_scaled_bounds)
    max_x = max(b[2] for b in all_scaled_bounds)
    min_y = min(b[1] for b in all_scaled_bounds)
    max_y = max(b[3] for b in all_scaled_bounds)
    logger.debug("Overall scaled extent: %.4f m × %.4f m", max_x - min_x, max_y - min_y)
    return updated


def smooth_geometry(contours: List[dict], smoothing: int) -> List[dict]:
    """Smooth projected contours using the buffer trick.

    Args:
        contours: Contours to smooth.
        smoothing: Buffer radius in metres; non-positive disables smoothing.

    Returns:
        List of smoothed contours.
    """
    if smoothing <= 0:
        return contours
    smoothed_contours = []
    for contour in contours:
        geom = shape(contour["geometry"])
        if smoothing > 0:
            geom = geom.buffer(smoothing).buffer(-smoothing)
        logger.debug(
            "Contour at elevation %s smoothed with radius %s",
            contour["elevation"],
            smoothing,
        )
        contour["geometry"] = mapping(geom)
        smoothed_contours.append(contour)
    return smoothed_contours


def filter_small_features(
    contours: List[dict], min_area_cm2: float, min_width_mm: float = 0.0
) -> List[dict]:
    """Remove tiny or thin features from contour layers.

    Args:
        contours: Contour list to filter.
        min_area_cm2: Minimum polygon area to keep in square centimetres.
        min_width_mm: Minimum width of features in millimetres.

    Returns:
        Filtered contour list.
    """
    min_area_m2 = min_area_cm2 / 1e4 if min_area_cm2 > 0 else 0.0
    min_width_m = min_width_mm / 1000.0 if min_width_mm > 0 else 0.0
    filtered = []
    for contour in contours:
        geom = shape(contour["geometry"])
        if min_width_m > 0:
            try:
                geom = geom.buffer(-min_width_m / 2).buffer(min_width_m / 2)
                geom = clean_geometry_strict(geom)
                if geom is None or geom.is_empty:
                    logger.debug(
                        "Contour @ %s filtered away (min width)",
                        contour.get("elevation"),
                    )
                    continue
            except Exception as exc:  # pragma: no cover - log and skip
                logger.warning("Buffer trick failed for min width filtering: %s", exc)
                continue
        parts = [g for g in _flatten_polygons([geom]) if g.area >= min_area_m2]
        if not parts:
            continue
        geom = MultiPolygon(parts)
        contour["geometry"] = mapping(geom)
        filtered.append(contour)
    return filtered
