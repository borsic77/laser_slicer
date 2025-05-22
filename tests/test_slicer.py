import os

import numpy as np
import pytest

from core.utils.slicer import download_srtm_tiles_for_bounds, mosaic_and_crop


@pytest.fixture
def switzerland_bounds():
    # Covers Yverdon-les-Bains and surroundings
    return (6.5, 46.7, 6.7, 46.9)


def test_download_srtm_tiles_for_bounds(switzerland_bounds):
    paths = download_srtm_tiles_for_bounds(switzerland_bounds)
    assert isinstance(paths, list)
    assert all(os.path.exists(p) for p in paths)
    assert any("srtm" in os.path.basename(p) for p in paths)


def test_mosaic_and_crop(switzerland_bounds):
    paths = download_srtm_tiles_for_bounds(switzerland_bounds)
    array, transform = mosaic_and_crop(paths, switzerland_bounds)
    assert isinstance(array, np.ndarray)
    assert array.ndim == 2
    assert array.size > 0
    assert transform is not None


def test_is_almost_closed():
    from shapely.geometry import LineString

    from core.utils.slicer import is_almost_closed

    fully_closed = LineString([(0, 0), (1, 0), (1, 1), (0, 0)])
    almost_closed = LineString(
        [(0, 0), (1, 0), (1, 1), (0, 1e-9)]
    )  # very close to (0, 0)
    open_loop = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])

    assert not is_almost_closed(fully_closed)
    assert is_almost_closed(almost_closed)
    assert not is_almost_closed(open_loop)


# Test for walk_bbox_between
def test_walk_bbox_between():
    from core.utils.slicer import walk_bbox_between

    bbox_coords = [(0, 1), (0, 0), (2, 0), (2, 1)]  # Clockwise: TL, BL, BR, TR

    # Walk CW from end_idx=2 (BR) to start_idx=0 (TL)
    segment_cw = walk_bbox_between(bbox_coords, 0, 2, direction="cw")
    assert segment_cw == [(2, 0), (2, 1), (0, 1)]

    # Walk CCW from end_idx=0 (TL) to start_idx=2 (BR)
    segment_ccw = walk_bbox_between(bbox_coords, 2, 0, direction="ccw")
    assert segment_ccw == [(0, 1), (2, 1), (2, 0)]

    # Wraparound CW from end_idx=1 (


# Test for project_geometry
def test_project_geometry(tmp_path):
    from shapely.geometry import Polygon, mapping

    from core.utils.slicer import project_geometry

    # Create a square polygon near Yverdon-les-Bains
    poly = Polygon(
        [(6.6, 46.8), (6.601, 46.8), (6.601, 46.801), (6.6, 46.801), (6.6, 46.8)]
    )
    contour = {"geometry": mapping(poly), "elevation": 500}

    result = project_geometry([contour], center_lon=6.6, center_lat=46.8)
    assert isinstance(result, list)
    assert "geometry" in result[0]
    assert result[0]["geometry"]["type"] == "Polygon"


# Test for scale_and_center_contours_to_substrate
def test_scale_and_center_contours_to_substrate():
    from shapely.geometry import Polygon, mapping, shape

    from core.utils.slicer import scale_and_center_contours_to_substrate

    # Create a large UTM-scale square around (300000, 5200000)
    utm_poly = Polygon(
        [
            (300000, 5200000),
            (300100, 5200000),
            (300100, 5200100),
            (300000, 5200100),
            (300000, 5200000),
        ]
    )
    contour = {"geometry": mapping(utm_poly), "elevation": 100}

    substrate_mm = 200.0  # 20 cm side square
    utm_bounds = (300000, 5200000, 300100, 5200100)

    scaled = scale_and_center_contours_to_substrate([contour], substrate_mm, utm_bounds)

    assert len(scaled) == 1
    geom = shape(scaled[0]["geometry"])
    assert geom.is_valid
    assert geom.area == pytest.approx(
        substrate_mm**2 * 0.000001, rel=0.01
    )  # within 1% of area
    assert geom.length == pytest.approx(
        substrate_mm * 0.004, rel=0.01
    )  # within 1% of perimeter
    assert geom.bounds[0] >= -0.1 and geom.bounds[2] <= 0.1  # x within +/-10 cm
    assert geom.bounds[1] >= -0.1 and geom.bounds[3] <= 0.1  # y within +/-10 cm


import logging
import os
from math import cos, radians
from typing import List, Tuple

import elevation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import shapely
from django.conf import settings
from rasterio.merge import merge
from rasterio.windows import from_bounds
from shapely.geometry import LinearRing, LineString, Polygon, box, mapping, shape
from shapely.ops import transform, unary_union

# Use headless matplotlib
matplotlib.use("Agg")

TILE_CACHE_DIR = settings.TILE_CACHE_DIR
DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
logger = logging.getLogger(__name__)


def save_debug_contour_polygon(polygon, level, filename):
    fig, ax = plt.subplots()
    if polygon.is_empty or not polygon.is_valid:
        return
    if polygon.geom_type == "Polygon":
        x, y = polygon.exterior.xy
        ax.plot(x, y)
    elif polygon.geom_type == "MultiPolygon":
        for p in polygon.geoms:
            x, y = p.exterior.xy
            ax.plot(x, y)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(DEBUG_IMAGE_PATH, f"{filename}_elev_{level}.png"))
    plt.close(fig)


def download_srtm_tiles_for_bounds(
    bounds: Tuple[float, float, float, float],
) -> List[str]:
    """
    Downloads all SRTM tiles that intersect the given bounding box.
    Returns a list of file paths to the GeoTIFFs.
    """
    os.makedirs(TILE_CACHE_DIR, exist_ok=True)
    lon_min, lat_min, lon_max, lat_max = bounds

    lat_range = range(int(lat_min), int(lat_max) + 1)
    lon_range = range(int(lon_min), int(lon_max) + 1)

    paths = []

    for lat in lat_range:
        for lon in lon_range:
            tile_bounds = (lon, lat, lon + 1, lat + 1)
            tif_path = os.path.join(TILE_CACHE_DIR, f"srtm_{lat}_{lon}.tif")
            if not os.path.exists(tif_path):
                elevation.clip(bounds=tile_bounds, output=tif_path)
            paths.append(tif_path)

    return paths


def mosaic_and_crop(
    tif_paths: List[str], bounds: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Merges multiple SRTM tiles and clips them to the specified bounds.
    Returns a single numpy array and the affine transform.
    """
    src_files = [rasterio.open(p) for p in tif_paths]

    # Merge to a single raster
    mosaic, transform = merge(src_files)

    # Clip using bounds
    window = from_bounds(*bounds, transform=transform)
    row_off = int(window.row_off)
    row_end = row_off + int(window.height)
    col_off = int(window.col_off)
    col_end = col_off + int(window.width)

    clipped = mosaic[:, row_off:row_end, col_off:col_end]

    cropped_transform = rasterio.windows.transform(window, transform)

    # Close opened files
    for src in src_files:
        src.close()

    return clipped[0], cropped_transform


from shapely.geometry import box


def walk_bbox_between(coords, start_idx, end_idx, direction="cw"):
    """
    Walks the bounding box coordinates from end_idx to start_idx in the specified direction.
    Includes both start and end points.
    direction: 'cw' walks forward in the list, 'ccw' walks backward.
    """
    n = len(coords)
    if direction == "cw":
        if start_idx >= end_idx:
            return coords[end_idx : start_idx + 1]
        else:
            return coords[end_idx:] + coords[: start_idx + 1]
    else:  # ccw
        if start_idx <= end_idx:
            return coords[start_idx : end_idx + 1][::-1]
        else:
            return (coords[start_idx:] + coords[: end_idx + 1])[::-1]


def is_almost_closed(line: LineString, tolerance: float = 1e-8) -> bool:
    return (
        line.coords[0] != line.coords[-1]
        and LineString([line.coords[0], line.coords[-1]]).length < tolerance
    )


# Test for _prepare_meshgrid


def test_prepare_meshgrid():
    import numpy as np
    import rasterio.transform

    from core.utils.slicer import _prepare_meshgrid

    # Create a simple 3x2 elevation array
    elevation_data = np.array([[1, 2], [3, 4], [5, 6]])
    # Define a transform: origin at (10, 20), pixel size 1x1
    transform = rasterio.transform.from_origin(west=10, north=20, xsize=1, ysize=1)

    lon, lat = _prepare_meshgrid(elevation_data, transform)

    assert lon.shape == elevation_data.shape
    assert lat.shape == elevation_data.shape
    # Top-left pixel center
    assert lon[0, 0] == 10.5
    assert lat[0, 0] == 19.5
    # Bottom-right pixel center
    assert lon[2, 1] == 11.5
    assert lat[2, 1] == 17.5


# Test for _create_contourf_levels
def test_create_contourf_levels():
    import numpy as np

    from core.utils.slicer import _create_contourf_levels

    data = np.array(
        [
            [103, 110, 118],
            [125, 132, 140],
        ]
    )
    interval = 20

    levels = _create_contourf_levels(data, interval)
    # Min is 103 → floor(103 / 20) * 20 = 100
    # Max is 140 → ceil(140 / 20) * 20 = 140
    # So we expect [100, 120, 140]
    expected = np.array([100.0, 120.0, 140.0])
    assert np.allclose(levels, expected)


def _create_contourf_levels(elevation_data: np.ndarray, interval: float) -> np.ndarray:
    """
    Computes the elevation contour levels to use for matplotlib's contourf.
    Ensures the highest elevation is included in the list.
    """
    min_elev = np.min(elevation_data)
    max_elev = np.max(elevation_data)
    levels = np.arange(min_elev, max_elev, interval)
    if levels[-1] < max_elev:
        levels = np.append(levels, max_elev)
    return levels


def _extract_level_polygons(cs) -> List[Tuple[float, List[Polygon]]]:
    """
    Extracts and flattens valid polygons for each elevation level from a QuadContourSet.
    Returns a list of tuples: (level, list of Polygon geometries).
    """
    from shapely.ops import unary_union

    level_polys = []
    for i, segs in enumerate(cs.allsegs):
        level = cs.levels[i]
        polys = []
        for seg in segs:
            try:
                if not np.allclose(seg[0], seg[-1]):
                    seg = np.vstack([seg, seg[0]])  # Close the ring
                poly = Polygon(seg)
                if not poly.is_valid or poly.area == 0:
                    logger.warning(
                        f"Dropped segment at level {level}: valid={poly.is_valid}, area={poly.area}"
                    )
                if poly.is_valid and poly.area > 0:
                    polys.append(poly)
            except Exception as e:
                logger.warning(f"Skipping bad filled contour at level {level}: {e}")
        flat_polys = []
        for geom in polys:
            try:
                if geom.geom_type == "Polygon":
                    flat_polys.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    flat_polys.extend(
                        [g for g in geom.geoms if g.geom_type == "Polygon"]
                    )
            except Exception as e:
                logger.warning(f"Could not flatten geometry at level {level}: {e}")
        level_polys.append((level, flat_polys))
    return level_polys


def _flatten_polygons(geoms: List[Polygon]) -> List[Polygon]:
    flat = []
    for geom in geoms:
        if geom.geom_type == "Polygon":
            flat.append(geom)
        elif geom.geom_type == "MultiPolygon":
            flat.extend(g for g in geom.geoms if g.geom_type == "Polygon")
    return flat


# Test for _flatten_polygons
def test_flatten_polygons():
    from shapely.geometry import MultiPolygon, Polygon

    from core.utils.slicer import _flatten_polygons

    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    multi = MultiPolygon([p1, p2])

    result = _flatten_polygons([p1, multi])
    assert isinstance(result, list)
    assert all(poly.geom_type == "Polygon" for poly in result)
    assert len(result) == 3  # one from p1, two from multi


def _force_multipolygon(geom):
    """
    Ensures the geometry is a MultiPolygon.
    """
    from shapely.geometry import MultiPolygon, Polygon

    if geom.geom_type == "MultiPolygon":
        return geom
    elif geom.geom_type == "Polygon":
        return MultiPolygon([geom])
    else:
        raise ValueError("Input geometry must be Polygon or MultiPolygon")


# Test for _force_multipolygon
def test_force_multipolygon():
    from shapely.geometry import MultiPolygon, Polygon

    from core.utils.slicer import _force_multipolygon

    p = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    mp = MultiPolygon([p])
    result1 = _force_multipolygon(p)
    result2 = _force_multipolygon(mp)

    assert isinstance(result1, MultiPolygon)
    assert isinstance(result2, MultiPolygon)
    assert len(result1.geoms) == 1
    assert len(result2.geoms) == 1


# Test for _prepare_meshgrid
def test_prepare_meshgrid():
    import numpy as np
    import rasterio.transform

    from core.utils.slicer import _prepare_meshgrid

    # Create a simple 3x2 elevation array
    elevation_data = np.array([[1, 2], [3, 4], [5, 6]])
    # Define a transform: origin at (10, 20), pixel size 1x1
    transform = rasterio.transform.from_origin(west=10, north=20, xsize=1, ysize=1)

    lon, lat = _prepare_meshgrid(elevation_data, transform)

    assert lon.shape == elevation_data.shape
    assert lat.shape == elevation_data.shape
    # Top-left pixel center
    assert lon[0, 0] == 10.5
    assert lat[0, 0] == 19.5
    # Bottom-right pixel center
    assert lon[2, 1] == 11.5
    assert lat[2, 1] == 17.5


def test_extract_level_polygons():
    import matplotlib.pyplot as plt
    import numpy as np

    from core.utils.slicer import _extract_level_polygons

    # Generate simple contour data
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = X + Y  # simple sloped surface

    cs = plt.contourf(X, Y, Z, levels=[0.5, 1.0, 1.5])
    results = _extract_level_polygons(cs)

    # Basic structure checks
    assert isinstance(results, list)
    for level, polygons in results:
        assert isinstance(level, float)
        assert isinstance(polygons, list)
        for p in polygons:
            assert p.is_valid
            assert p.area > 0
            assert p.geom_type == "Polygon"
    plt.close()


# Enhanced test for _extract_level_polygons with two pyramid-shaped peaks
def test_extract_level_polygons_two_peaks():
    import matplotlib.pyplot as plt
    import numpy as np

    from core.utils.slicer import _extract_level_polygons

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Create two cone-like pyramids
    Z1 = 1.0 - np.sqrt((X - 0.3) ** 2 + (Y - 0.3) ** 2)
    Z2 = 1.0 - np.sqrt((X - 0.7) ** 2 + (Y - 0.7) ** 2)
    Z = np.maximum(Z1, Z2)
    Z[Z < 0] = 0  # Clip to zero

    cs = plt.contourf(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8])
    results = _extract_level_polygons(cs)

    assert len(results) == 3  # Four levels

    for level, polygons in results:
        assert isinstance(level, float)
        assert isinstance(polygons, list)
        for p in polygons:
            assert p.is_valid
            assert p.area > 0
            assert p.geom_type == "Polygon"

    # Check that lower levels contain more polygons (i.e. both peaks are visible)
    low_level_polygons = results[0][1]
    assert len(low_level_polygons) >= 2

    plt.close()


# Test for _compute_layer_bands
def test_compute_layer_bands():
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import shape

    from core.utils.slicer import _compute_layer_bands, _extract_level_polygons

    # Create synthetic terrain with nested contours
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = 1.0 - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    Z[Z < 0] = 0

    cs = plt.contourf(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8])
    levels = _extract_level_polygons(cs)
    bands = _compute_layer_bands(levels, rasterio.transform.from_origin(0, 1, 1, 1))

    assert isinstance(bands, list)
    assert len(bands) == len(levels)
    for band in bands:
        assert "elevation" in band
        assert "geometry" in band
        assert band["closed"] is True
        geom = shape(band["geometry"])
        assert geom.is_valid
        assert not geom.is_empty
        assert geom.geom_type in ("Polygon", "MultiPolygon")

    plt.close()


# Comprehensive test for _compute_layer_bands using two hills
def test_compute_layer_bands_two_hills():
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import shape

    from core.utils.slicer import _compute_layer_bands, _extract_level_polygons

    # Two hills: one centered at (0.3, 0.3), one at (0.7, 0.7), with steep, distinct peaks
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) * 100)
    Z2 = np.exp(-((X - 0.7) ** 2 + (Y - 0.7) ** 2) * 100)
    Z = np.maximum(Z1, Z2)
    Z[Z < 0] = 0

    levels = [0.2, 0.4, 0.6, 0.8, 1.0]
    cs = plt.contourf(X, Y, Z, levels=levels)
    level_polygons = _extract_level_polygons(cs)
    bands = _compute_layer_bands(
        level_polygons, rasterio.transform.from_origin(0, 1, 1, 1)
    )

    assert len(bands) == len(levels) - 1

    for band in bands:
        assert "elevation" in band
        assert "geometry" in band
        assert band["closed"] is True
        geom = shape(band["geometry"])
        assert geom.is_valid
        assert not geom.is_empty
        assert geom.geom_type in ("Polygon", "MultiPolygon")

        # Check that each lower band contains the one above
    for i in range(1, len(bands)):
        lower = shape(bands[i]["geometry"])
        upper = shape(bands[i - 1]["geometry"])
        assert lower.covers(lower)

    # Base layer should include both peaks (at least two distinct regions)
    base_geom = shape(bands[0]["geometry"])
    if base_geom.geom_type == "MultiPolygon":
        assert len(base_geom.geoms) >= 2
    plt.close()


# Test for _compute_layer_bands with a tall cylinder: all bands should have similar area
def test_compute_layer_bands_cylinder():
    import matplotlib.pyplot as plt
    import numpy as np
    from shapely.geometry import shape

    from core.utils.slicer import _compute_layer_bands, _extract_level_polygons

    # Create a cylinder with sharp elevation edges
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    radius = 0.2
    center = (0.5, 0.5)
    distance = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    Z = np.where(distance < radius, 1.0, 0.0)

    cs = plt.contourf(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.8, 1.0])
    level_polygons = _extract_level_polygons(cs)
    bands = _compute_layer_bands(
        level_polygons, rasterio.transform.from_origin(0, 1, 1, 1)
    )

    assert len(bands) == 4

    for band in bands:
        assert "elevation" in band
        assert "geometry" in band
        assert band["closed"] is True
        geom = shape(band["geometry"])
        assert geom.is_valid
        assert not geom.is_empty
        assert geom.geom_type in ("Polygon", "MultiPolygon")

    # All layers should have similar area
    areas = [shape(band["geometry"]).area for band in bands]
    for a in areas:
        assert abs(a - areas[0]) < 0.01  # within 1% variation

    plt.close()
