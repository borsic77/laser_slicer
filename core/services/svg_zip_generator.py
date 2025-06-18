from __future__ import annotations

import io
import zipfile
from typing import Iterable

from core.utils.svg_export import contours_to_svg_zip

__all__ = ["generate_svg_layers", "zip_svgs"]


# just reuse old code, the files are small so the overhead of unzipping and rezipping is small
# and we don't need to change the SVG generation code
def generate_svg_layers(
    contours: list[dict],
    *,
    basename: str = "contours",
    stroke_cut: str = "#000000",
    stroke_align: str = "#ff0000",
    stroke_road: str = "#0000ff",
    stroke_building: str = "#880000",
    stroke_width_mm: float = 0.1,
) -> list[tuple[str, bytes]]:
    """
    Convert *contour bands* into **separate SVG files**.

    Parameters
    ----------
    contours
        Output of ContourSlicingJob.run() – list of dicts with 'geometry', …
    basename
        Base filename (without extension) for every layer.
    stroke_cut, stroke_align, stroke_width_mm
        Passed straight through to the low-level exporter.

    Returns
    -------
    list[(str, bytes)]
        `[("layer_00_contours.svg", b"..."), ...]`
    """
    # Reuse the existing exporter so the SVG drawing itself is unchanged.
    zip_bytes: bytes = contours_to_svg_zip(
        contours,
        basename=basename,
        stroke_cut=stroke_cut,
        stroke_align=stroke_align,
        stroke_width_mm=stroke_width_mm,
        stroke_road=stroke_road,
        stroke_building=stroke_building,
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return [(name, zf.read(name)) for name in zf.namelist()]


def zip_svgs(svg_files: Iterable[tuple[str, bytes]]) -> bytes:
    """
    Bundle individual SVG files into one in-memory ZIP archive.

    Parameters
    ----------
    svg_files
        Iterable of `(filename, raw_svg_bytes)`.

    Returns
    -------
    bytes
        A ready-to-stream ZIP archive.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in svg_files:
            zf.writestr(fname, data)
    buffer.seek(0)
    return buffer.getvalue()
