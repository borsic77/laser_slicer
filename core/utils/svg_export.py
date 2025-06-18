"""
svg_export.py

Generates laser-cutter-ready SVG files from a list of elevation contour bands,
and bundles them as a ZIP archive.

Each SVG encodes one contour layer (for cutting) and, optionally, an alignment
outline for stacking, plus a label. Coordinates are assumed to be in UTM (meters)
and are mapped to SVG millimeters with Y-axis flipped to match physical orientation.

Intended for use with laser-cuttable topographic models.
"""

import io
import math
import zipfile
from typing import List, Tuple

import svgwrite
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape


def _iter_polygons(geom):
    """
    Yield each polygon in geom (handles both Polygon and MultiPolygon).
    Args:
        geom (BaseGeometry): Shapely geometry (Polygon or MultiPolygon).
    Yields:
        Polygon: Each polygon in the geometry."""
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            yield poly


def linestring_to_svg_path(
    ls,
    glob_minx: float,
    glob_maxx: float,
    glob_miny: float,
    glob_maxy: float,
) -> str:
    """
    Convert a LineString or MultiLineString to an SVG path string.
    Args:
        ls (LineString or MultiLineString): Shapely LineString or MultiLineString.
        glob_minx (float): Global minimum X coordinate.
        glob_maxx (float): Global maximum X coordinate.
        glob_miny (float): Global minimum Y coordinate.
        glob_maxy (float): Global maximum Y coordinate.
    Returns:
        str: SVG path 'd' attribute string.
    """

    def _ls_to_cmds(coords):
        if not coords:
            return ""
        cmds = [
            "M {:.3f} {:.3f}".format(
                *_to_svg_coords(*coords[0], glob_minx, glob_maxx, glob_miny, glob_maxy)
            )
        ]
        for x, y in coords[1:]:
            cmds.append(
                "L {:.3f} {:.3f}".format(
                    *_to_svg_coords(x, y, glob_minx, glob_maxx, glob_miny, glob_maxy)
                )
            )
        return " ".join(cmds)

    if isinstance(ls, LineString):
        return _ls_to_cmds(list(ls.coords))
    elif isinstance(ls, MultiLineString):
        return " ".join(_ls_to_cmds(list(geom.coords)) for geom in ls.geoms)
    else:
        return ""


__all__ = [
    "contours_to_svg_zip",
]


def remove_holes(geom):
    """
    Remove holes from a Shapely geometry.
    Args:
        geom (BaseGeometry): Shapely geometry (Polygon or MultiPolygon).
    Returns:
        BaseGeometry: Geometry with holes removed.
    """
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        # Keep only the exterior
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(poly.exterior) for poly in geom.geoms])
    else:
        return geom  # Unchanged for other types


def _to_svg_coords(
    x_m: float,
    y_m: float,
    glob_minx: float,
    glob_maxx: float,
    glob_miny: float,
    glob_maxy: float,
) -> Tuple[float, float]:
    """
    Convert UTM coordinates (in meters) to SVG coordinates (in mm), flipping
    Y axis to map north-up (UTM) to SVG’s y-down.

    Args:
        x_m (float): X coordinate in UTM meters.
        y_m (float): Y coordinate in UTM meters.
        glob_minx (float): Global minimum X coordinate.
        glob_maxx (float): Global maximum X coordinate.
        glob_miny (float): Global minimum Y coordinate.
        glob_maxy (float): Global maximum Y coordinate.

    Returns:
        Tuple[float, float]: Corresponding X, Y in SVG mm coordinates.
    """
    x_mm = (x_m - glob_minx) * 1000.0
    y_mm = (glob_maxy - y_m) * 1000.0
    return round(x_mm, 3), round(y_mm, 3)


def _polygon_to_path(
    p: Polygon,
    glob_minx: float,
    glob_maxx: float,
    glob_miny: float,
    glob_maxy: float,
    include_holes: bool = True,
) -> str:
    """Convert a Shapely Polygon to an SVG path string.
    Close path with ‘Z’ so laser doesn’t leave open segments.

    Args:
        p (Polygon): Polygon to convert.
        glob_minx (float): Global minimum X coordinate.
        glob_maxx (float): Global maximum X coordinate.
        glob_miny (float): Global minimum Y coordinate.
        glob_maxy (float): Global maximum Y coordinate.
        include_holes (bool): Whether to include interior rings.

    Returns:
        str: SVG path 'd' attribute string.
    """

    def _ring_to_cmds(coords):
        if not coords:
            return ""
        cmds = [
            "M {:.3f} {:.3f}".format(
                *_to_svg_coords(*coords[0], glob_minx, glob_maxx, glob_miny, glob_maxy)
            )
        ]
        for x, y in coords[1:]:
            cmds.append(
                "L {:.3f} {:.3f}".format(
                    *_to_svg_coords(x, y, glob_minx, glob_maxx, glob_miny, glob_maxy)
                )
            )
        cmds.append("Z")
        return " ".join(cmds)

    outer = _ring_to_cmds(list(p.exterior.coords))
    inners = ""
    if include_holes:
        inners = " ".join(_ring_to_cmds(list(r.coords)) for r in p.interiors)
    return f"{outer} {inners}".strip()


def _geom_to_paths(
    g,
    glob_minx: float,
    glob_maxx: float,
    glob_miny: float,
    glob_maxy: float,
    include_holes: bool = True,
) -> List[str]:
    """Convert a Shapely geometry into a list of SVG path strings.
    Recursively decompose MultiPolygons.

    Args:
        g (BaseGeometry): Shapely geometry (Polygon or MultiPolygon).
        glob_minx (float): Global minimum X coordinate.
        glob_maxx (float): Global maximum X coordinate.
        glob_miny (float): Global minimum Y coordinate.
        glob_maxy (float): Global maximum Y coordinate.
        include_holes (bool): Whether to include holes in the path.

    Returns:
        List[str]: List of SVG path strings.
    """
    if g.geom_type == "Polygon":
        return [
            _polygon_to_path(
                g,
                glob_minx,
                glob_maxx,
                glob_miny,
                glob_maxy,
                include_holes=include_holes,
            )
        ]
    elif g.geom_type == "MultiPolygon":
        paths = []
        for poly in g.geoms:
            paths.extend(
                _geom_to_paths(
                    poly,
                    glob_minx,
                    glob_maxx,
                    glob_miny,
                    glob_maxy,
                    include_holes=include_holes,
                )
            )
        return paths
    else:
        return []


def contours_to_svg_zip(
    contours: List[dict],
    stroke_cut: str = "#000000",
    stroke_align: str = "#ff0000",
    stroke_road: str = "#0000ff",
    stroke_building: str = "#880000",
    stroke_width_mm: float = 0.1,  # may need to be increased for some lasers
    basename: str = "contours",
) -> bytes:
    """
    Convert a list of contour *bands* (as returned by
    ``generate_contours`` / ``scale_and_center_contours_to_substrate``)
    into a ZIP archive of individual‑layer SVG files. Bands are assumed to be
    Shapely geometries with a "geometry" key containing the actual geometry
    (e.g., a Polygon or MultiPolygon) and an "elevation" key with the
    corresponding elevation value. The bands are intended for use in
    laser-cutting applications, where each layer represents a slice of
    topography at a specific elevation.
    Each layer is represented by a separate SVG file in the ZIP archive.

    Each SVG contains:
    1. *cut* geometry for the current layer ("stroke_cut" colour).
    2. Outline of the layer **above** it ("stroke_align" colour) for alignment.
    3. For each polygon in the cut geometry, a label of the form "#<layer>_<poly>"
       is drawn. The label is placed inside the alignment geometry (if the centroid
       of the cut polygon is covered by any polygon in the alignment geometry),
       and colored green. If not covered by any alignment polygon, the label is
       placed in the cut polygon and colored blue. This ensures labels are always
       visible and indicate whether the cut polygon is covered by the alignment outline.
       (See inline comments for details.)

    Geometry is assumed to be projected in **metres** (UTM).  SVG coordinates
    are expressed in millimetres with the *Y axis flipped*
    so that the visual orientation in the browser matches the physical model
    (SVG Y grows down, UTM north is positive up).
    Args:
        contours (List[dict]): List of contour geometries with elevation, assumed to be in UTM metres.
        stroke_cut (str): Stroke color for the cut geometry (default: black).
        stroke_align (str): Stroke color for the alignment geometry (default: red).
        stroke_road (str): Stroke color for road geometries.
        stroke_building (str): Stroke color for building outlines.
        stroke_width_mm (float): Width of strokes in millimeters (default: 0.1 mm).
        basename (str): Base filename used for SVG layers in the ZIP.
    Returns:
        bytes: The ZIP file content as bytes.
    """

    # Processing steps per layer:
    # 1. Draw the cut path (material to cut for this layer)
    # 2. Draw text labels (hidden or visible depending on stacking)
    # 3. Draw alignment outline (non-overlapping segments only)

    if not contours:
        raise ValueError("Contours list must not be empty")

    # ------------------------------------------------------------------
    # Establish a *global* bounding box so every layer shares the same
    # coordinate system → every cut aligns perfectly when imported into a
    # laser‑cutter UI.
    # ------------------------------------------------------------------
    xs, ys = [], []
    for c in contours:
        minx, miny, maxx, maxy = shape(c["geometry"]).bounds
        xs += [minx, maxx]
        ys += [miny, maxy]

    glob_minx, glob_maxx = min(xs), max(xs)
    glob_miny, glob_maxy = min(ys), max(ys)

    width_mm = (glob_maxx - glob_minx) * 1000.0
    height_mm = (glob_maxy - glob_miny) * 1000.0

    # ---------------------------------------------------------------
    # Build the ZIP in‑memory
    # The "cut" geometry is the unique material of this layer (difference with the layer above).
    # The "align" geometry is only the outer shell of the next layer up, for manual stacking alignment.
    # ---------------------------------------------------------------
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        n_layers = len(contours)
        for idx, layer in enumerate(contours):
            band_i = shape(layer["geometry"])
            band_above = (
                shape(contours[idx + 1]["geometry"]) if idx + 1 < n_layers else None
            )

            # *Cut* geometry: the material for *this* slice only
            cut_geom = remove_holes(band_i)

            # Alignment outline: use band_above
            align_geom_rough = (
                remove_holes(band_above) if band_above is not None else None
            )

            align_geom = align_geom_rough

            # Compute the alignment outline to engrave: only those outline segments
            # of align_geom that do not coincide with cut_geom's boundary.
            # Subtracting boundaries ensures we do not engrave alignment lines where the laser will already cut, preventing duplicate scoring.”
            engrave_outline = None
            if align_geom is not None and not align_geom.is_empty:
                # Difference of boundaries: align_geom.boundary - cut_geom.boundary
                engrave_outline = align_geom.boundary.difference(cut_geom.boundary)

            dwg = svgwrite.Drawing(
                size=(f"{width_mm:.3f}mm", f"{height_mm:.3f}mm"),
                viewBox=f"0 0 {width_mm:.3f} {height_mm:.3f}",
            )
            dwg.set_desc(title=f"Elevation: {layer['elevation']} m, Layer: #{idx + 1}")

            # ----------------------------------------------
            # Cut paths – solid stroke
            # ----------------------------------------------
            for path_d in _geom_to_paths(
                cut_geom, glob_minx, glob_maxx, glob_miny, glob_maxy
            ):
                dwg.add(
                    dwg.path(
                        d=path_d,
                        stroke=stroke_cut,
                        fill="none",
                        stroke_width=f"{stroke_width_mm:.3f}mm",
                    )
                )

            # ----------------------------------------------
            # Label placement and color logic:
            # - For each polygon in the cut layer, check if it overlaps any alignment polygon above.
            # - If so, place label inside the overlap area (hidden after stacking), color green,
            #   and use the *area of the intersection* to determine label/font size.
            # - If not, place label in the cut polygon itself (visible after stacking), color blue,
            #   and use the *area of the cut polygon* for sizing.
            # - All labels are drawn beneath alignment outlines if possible for assembly guidance.
            # ----------------------------------------------
            align_polys = []
            if align_geom is not None and not align_geom.is_empty:
                # Flatten alignment geometry to list of polygons
                if isinstance(align_geom, Polygon):
                    align_polys = [align_geom]
                elif isinstance(align_geom, MultiPolygon):
                    align_polys = list(align_geom.geoms)
                else:
                    # If alignment geometry is other type, ignore for label placement
                    align_polys = []

            for i, cut_poly in enumerate(_iter_polygons(cut_geom)):
                area = cut_poly.area
                if area == 0.0:
                    continue
                label_text = f"#{idx + 1}_{i + 1}"
                label_pt = None
                label_color = "#0000cc"  # blue: visible in assembled model
                char_size = math.sqrt(area)
                # Try to find an intersection with any alignment polygon
                for ap in align_polys:
                    intersection = cut_poly.intersection(ap)
                    if not intersection.is_empty and intersection.area > 0.0:
                        # Overlap found: use intersection area for sizing
                        label_pt = intersection.representative_point()
                        label_color = "#00aa00"  # green: hidden in assembled model
                        char_size = math.sqrt(intersection.area)
                        break
                if label_pt is None:
                    # No overlap: place label in cut polygon, color blue, use cut_poly area for sizing
                    label_pt = cut_poly.representative_point()
                # Font size: scale factor, clamp between 2mm and 10mm
                font_size = max(2.0, min(0.15 * char_size * 1000, 10.0))  # 1000 to mm
                lx, ly = _to_svg_coords(
                    label_pt.x, label_pt.y, glob_minx, glob_maxx, glob_miny, glob_maxy
                )
                dwg.add(
                    dwg.text(
                        label_text,
                        insert=(lx, ly),
                        fill=label_color,
                        font_size=f"{font_size:.2f}mm",
                        style="font-family:monospace;text-anchor:middle;dominant-baseline:middle;",
                    )
                )

            # ----------------------------------------------
            # Alignment outline – secondary colour
            # Only engrave outline segments of align_geom that do not coincide with cut_geom's boundary.
            # engrave_outline may be empty, a LineString, MultiLineString, or GeometryCollection.
            # Draw each LineString/MultiLineString segment.
            # ----------------------------------------------
            if engrave_outline is not None and not engrave_outline.is_empty:
                # engrave_outline can be LineString, MultiLineString, or GeometryCollection
                outlines = []
                if isinstance(engrave_outline, (LineString, MultiLineString)):
                    outlines = [engrave_outline]
                elif hasattr(engrave_outline, "geoms"):
                    # GeometryCollection or MultiLineString
                    outlines = [
                        g
                        for g in engrave_outline.geoms
                        if isinstance(g, (LineString, MultiLineString))
                    ]
                for outline in outlines:
                    if isinstance(outline, LineString):
                        path_d = linestring_to_svg_path(
                            outline, glob_minx, glob_maxx, glob_miny, glob_maxy
                        )
                        if path_d.strip():
                            dwg.add(
                                dwg.path(
                                    d=path_d,
                                    stroke=stroke_align,
                                    fill="none",
                                    stroke_width=f"{stroke_width_mm:.3f}mm",
                                )
                            )
                    elif isinstance(outline, MultiLineString):
                        for ls in outline.geoms:
                            path_d = linestring_to_svg_path(
                                ls, glob_minx, glob_maxx, glob_miny, glob_maxy
                            )
                            if path_d.strip():
                                dwg.add(
                                    dwg.path(
                                        d=path_d,
                                        stroke=stroke_align,
                                        fill="none",
                                        stroke_width=f"{stroke_width_mm:.3f}mm",
                                    )
                                )

            # ----------------------------------------------
            # Optional road and building geometries
            # ----------------------------------------------
            if layer.get("roads") is not None:
                road_geom = shape(layer["roads"])
                if not road_geom.is_empty:
                    segments = [road_geom] if isinstance(road_geom, LineString) else getattr(road_geom, "geoms", [road_geom])
                    for seg in segments:
                        path_d = linestring_to_svg_path(seg, glob_minx, glob_maxx, glob_miny, glob_maxy)
                        if path_d.strip():
                            dwg.add(
                                dwg.path(
                                    d=path_d,
                                    stroke=stroke_road,
                                    fill="none",
                                    stroke_width=f"{stroke_width_mm:.3f}mm",
                                )
                            )
            if layer.get("buildings") is not None:
                build_geom = shape(layer["buildings"])
                if not build_geom.is_empty:
                    for path_d in _geom_to_paths(
                        build_geom, glob_minx, glob_maxx, glob_miny, glob_maxy, include_holes=False
                    ):
                        dwg.add(
                            dwg.path(
                                d=path_d,
                                stroke=stroke_building,
                                fill="none",
                                stroke_width=f"{stroke_width_mm:.3f}mm",
                            )
                        )

            svg_bytes = dwg.tostring().encode()
            # Filename encodes layer order and elevation for traceability in multi-layer jobs.
            fname = (
                f"layer_{idx + 1:02d}_{int(round(layer['elevation']))}m_{basename}.svg"
            )
            zf.writestr(fname, svg_bytes)

    mem_zip.seek(0)
    return mem_zip.getvalue()
