import io
import zipfile
from typing import Dict, List, Tuple

import svgwrite
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from core.utils.export_filename import build_export_basename

__all__ = [
    "contours_to_svg_zip",
]


# -----------------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------------


def contours_to_svg_zip(
    contours: List[dict],
    stroke_cut: str = "#000000",
    stroke_align: str = "#ff0000",
    stroke_width_mm: float = 0.1,  # may need to be increased for some lasers
    basename: str = "contours",
) -> bytes:
    """Convert a list of contour *bands* (as returned by
    ``generate_contours`` / ``scale_and_center_contours_to_substrate``)
    into a ZIP archive of individual‑layer SVG files.

    Each SVG contains
    1. *cut* geometry for the current layer ("stroke_cut" colour).
    2. Outline of the layer **above** it ("stroke_align" colour) for alignment.
    3. A text label with "<elevation> / #<index>" centred inside the alignment
       outline (or the cut outline for the top layer).

    Geometry is assumed to be projected in **metres** (UTM).  SVG coordinates
    are expressed in millimetres with the *Y axis flipped*
    so that the visual orientation in the browser matches the physical model
    (SVG Y grows down, UTM north is positive up).
    Args:
        contours (List[dict]): List of contour geometries with elevation, assumed to be in UTM metres.
        stroke_cut (str): Stroke color for the cut geometry (default: black).
        stroke_align (str): Stroke color for the alignment geometry (default: red).
        stroke_width_mm (float): Width of strokes in millimeters (default: 0.1 mm).
        basename (str): Base filename used for SVG layers in the ZIP.
    Returns:
        bytes: The ZIP file content as bytes.
    """

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

    width_mm = glob_maxx - glob_minx
    height_mm = glob_maxy - glob_miny

    def _to_svg_coords(x_m: float, y_m: float) -> Tuple[float, float]:
        """Convert UTM coordinates (in meters) to SVG coordinates (in mm), flipping Y.

        Args:
            x_m (float): X coordinate in UTM meters.
            y_m (float): Y coordinate in UTM meters.

        Returns:
            Tuple[float, float]: Corresponding X, Y in SVG mm coordinates.
        """
        x_mm = (x_m - glob_minx) * 1000.0
        y_mm = (glob_maxy - y_m) * 1000.0
        return round(x_mm, 3), round(y_mm, 3)

    def _polygon_to_path(p: Polygon, include_holes: bool = True) -> str:
        """Convert a Shapely Polygon to an SVG path string.

        Args:
            p (Polygon): Polygon to convert.
            include_holes (bool): Whether to include interior rings.

        Returns:
            str: SVG path 'd' attribute string.
        """

        def _ring_to_cmds(coords):
            if not coords:
                return ""
            cmds = ["M {:.3f} {:.3f}".format(*_to_svg_coords(*coords[0]))]
            for x, y in coords[1:]:
                cmds.append("L {:.3f} {:.3f}".format(*_to_svg_coords(x, y)))
            cmds.append("Z")
            return " ".join(cmds)

        outer = _ring_to_cmds(list(p.exterior.coords))
        inners = ""
        if include_holes:
            inners = " ".join(_ring_to_cmds(list(r.coords)) for r in p.interiors)
        return f"{outer} {inners}".strip()

    def _geom_to_paths(g, include_holes: bool = True) -> List[str]:
        """Convert a Shapely geometry into a list of SVG path strings.

        Args:
            g (BaseGeometry): Shapely geometry (Polygon or MultiPolygon).
            include_holes (bool): Whether to include holes in the path.

        Returns:
            List[str]: List of SVG path strings.
        """
        if g.geom_type == "Polygon":
            return [_polygon_to_path(g, include_holes=include_holes)]
        elif g.geom_type == "MultiPolygon":
            paths = []
            for poly in g.geoms:
                paths.extend(_geom_to_paths(poly, include_holes=include_holes))
            return paths
        else:
            return []

    # ---------------------------------------------------------------
    # Build the ZIP in‑memory
    # ---------------------------------------------------------------
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        n_layers = len(contours)
        for idx, layer in enumerate(contours):
            band_i = shape(layer["geometry"])
            band_above = (
                shape(contours[idx + 1]["geometry"]) if idx + 1 < n_layers else None
            )
            band_above_above = (
                shape(contours[idx + 2]["geometry"]) if idx + 2 < n_layers else None
            )

            # *Cut* geometry: the material for *this* slice only
            cut_geom = (
                band_i.difference(band_above).buffer(0)
                if band_above is not None and not band_above.is_empty
                else band_i
            )
            # Alignment geometry: only use the single layer above
            if band_above and band_above_above:
                align_geom = band_above.difference(band_above_above).buffer(0)
            else:
                align_geom = band_above

            dwg = svgwrite.Drawing(
                size=(f"{width_mm:.3f}mm", f"{height_mm:.3f}mm"),
                viewBox=f"0 0 {width_mm:.3f} {height_mm:.3f}",
            )

            # ----------------------------------------------
            # Cut paths – solid stroke
            # ----------------------------------------------
            for path_d in _geom_to_paths(cut_geom):
                dwg.add(
                    dwg.path(
                        d=path_d,
                        stroke=stroke_cut,
                        fill="none",
                        stroke_width=f"{stroke_width_mm:.3f}mm",
                    )
                )

            # ----------------------------------------------
            # Alignment outline – secondary colour
            # ----------------------------------------------
            if align_geom is not None and not align_geom.is_empty:
                for path_d in _geom_to_paths(align_geom, include_holes=False):
                    dwg.add(
                        dwg.path(
                            d=path_d,
                            stroke=stroke_align,
                            fill="none",
                            stroke_width=f"{stroke_width_mm:.3f}mm",
                        )
                    )
            label = f"{layer['elevation']} / #{idx + 1}"
            # Place label(s) inside each polygon of the alignment geometry (or cut geometry if top layer)
            target_geom = align_geom if align_geom is not None else band_i
            geoms = (
                [target_geom]
                if target_geom.geom_type == "Polygon"
                else list(target_geom.geoms)
            )
            for poly in geoms:
                label_point = poly.representative_point()
                lx, ly = _to_svg_coords(label_point.x, label_point.y)
                dwg.add(
                    dwg.text(
                        label,
                        insert=(lx, ly),
                        text_anchor="middle",
                        alignment_baseline="central",
                        font_size="8pt",
                        fill=stroke_align,
                    )
                )

            svg_bytes = dwg.tostring().encode()
            fname = (
                f"layer_{idx + 1:02d}_{int(round(layer['elevation']))}m_{basename}.svg"
            )
            zf.writestr(fname, svg_bytes)

    mem_zip.seek(0)
    return mem_zip.getvalue()
