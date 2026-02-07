import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.django_settings_stub")

import pytest
from shapely.affinity import rotate
from shapely.geometry import Polygon, box, mapping

from core.services.svg_zip_generator import generate_svg_layers, zip_svgs
from core.utils.geometry_ops import (
    _grid_convergence_angle_from_geometry,
    clip_contours_to_bbox,
)


def test_grid_convergence_angle_simple():
    rect = box(0, 0, 1, 4)
    rotated = rotate(rect, 30, origin="center", use_radians=False)
    angle = _grid_convergence_angle_from_geometry([rotated])
    assert pytest.approx(angle, abs=1e-2) == 30


def test_grid_convergence_angle_correction():
    rect = box(0, 0, 1, 4)
    rotated = rotate(rect, 60, origin="center", use_radians=False)
    angle = _grid_convergence_angle_from_geometry([rotated])
    # Angle > 45° should be corrected to keep north mostly up
    assert pytest.approx(angle, abs=1e-2) == 60


def test_clip_contours_to_bbox():
    inside = box(1, 1, 3, 3)
    partial = box(2, 2, 5, 5)
    outside = box(6, 6, 8, 8)
    contours = [
        {"geometry": mapping(inside), "elevation": 0},
        {"geometry": mapping(partial), "elevation": 1},
        {"geometry": mapping(outside), "elevation": 2},
    ]
    clipped = clip_contours_to_bbox(contours, (0, 0, 4, 4))
    assert len(clipped) == 2
    polys = [
        Polygon(c["geometry"]["coordinates"][0])
        if c["geometry"]["type"] == "Polygon"
        else None
        for c in clipped
    ]
    assert all(p.bounds[0] >= 0 and p.bounds[2] <= 4 for p in polys)


def test_generate_and_zip_svgs(tmp_path):
    poly1 = box(0, 0, 1, 1)
    poly2 = box(0, 0, 1, 2)
    contours = [
        {"geometry": mapping(poly1), "elevation": 0},
        {"geometry": mapping(poly2), "elevation": 1},
    ]
    svgs = generate_svg_layers(contours, basename="test")
    assert len(svgs) == 2
    assert all(
        name.endswith(".svg") and data.startswith(b"<svg") for name, data in svgs
    )

    zip_bytes = zip_svgs(svgs)
    zip_path = tmp_path / "out.zip"
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    assert zip_path.exists() and zip_path.stat().st_size > 0
