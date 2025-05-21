import io
import logging
import traceback
from functools import wraps

import numpy as np
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from pyproj import Transformer
from rest_framework.decorators import api_view
from rest_framework.response import Response

from core.utils.export_filename import build_export_basename
from core.utils.geocoding import geocode_address
from core.utils.slicer import (
    download_srtm_tiles_for_bounds,
    generate_contours,
    mosaic_and_crop,
    project_geometry,
    scale_and_center_contours_to_substrate,
)
from core.utils.svg_export import contours_to_svg_zip

logger = logging.getLogger(__name__)


def safe_api(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        try:
            return view_func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"API failure in {view_func.__name__}")
            return Response({"error": "An unexpected error occurred."}, status=500)

    return wrapper


@csrf_exempt
@api_view(["POST"])
@safe_api
def elevation_range(request) -> Response:
    """Return the min and max elevation in the given bounding box.

    Args:
        request (Request): Django REST Framework request containing bounds (lat/lon).

    Returns:
        Response: JSON response with 'min' and 'max' elevation values.
    """
    bounds = request.data["bounds"]

    lat_min = float(bounds["lat_min"])
    lon_min = float(bounds["lon_min"])
    lat_max = float(bounds["lat_max"])
    lon_max = float(bounds["lon_max"])

    tile_paths = download_srtm_tiles_for_bounds((lon_min, lat_min, lon_max, lat_max))
    elevation, _ = mosaic_and_crop(tile_paths, (lon_min, lat_min, lon_max, lat_max))

    masked = np.ma.masked_where(
        ~np.isfinite(elevation) | (elevation <= -32768), elevation
    )
    min_elev = max(-500, float(masked.min()))
    max_elev = min(10000, float(masked.max()))

    return Response({"min": min_elev, "max": max_elev})


@csrf_exempt
@api_view(["POST"])
@safe_api
def export_svgs(request) -> FileResponse | Response:
    """Generate a ZIP archive of SVG files from provided contour layers.

    Args:
        request (Request): Django REST Framework request containing layers and metadata.

    Returns:
        FileResponse | Response: A ZIP file response or error message.
    """
    contours = request.data.get("layers")  # front-end sends the list it already has
    if not contours:
        return Response({"error": "No layers supplied"}, status=400)

    address = request.data.get("address", "").strip()
    coords = request.data.get("coordinates", None)
    height_mm = request.data.get("height_per_layer", "unknown")
    num_layers = len(contours)
    logger.debug(
        f"Exporting {num_layers} layers for address: {address}, coords: {coords}, height: {height_mm}"
    )

    base_filename = build_export_basename(address, coords, height_mm, num_layers)
    filename = f"{base_filename}.zip"

    zip_bytes = contours_to_svg_zip(
        contours, basename=base_filename
    )  # generate SVGs in memory

    response = FileResponse(
        io.BytesIO(zip_bytes),
        content_type="application/zip",
        as_attachment=True,
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    response["Access-Control-Expose-Headers"] = "Content-Disposition"
    return response


def compute_utm_bounds_from_wgs84(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    center_x: float,
    center_y: float,
) -> tuple[float, float, float, float]:
    """Convert WGS84 bounding box to projected UTM bounds.

    Args:
        lon_min (float): Minimum longitude of bounding box.
        lat_min (float): Minimum latitude of bounding box.
        lon_max (float): Maximum longitude of bounding box.
        lat_max (float): Maximum latitude of bounding box.
        center_x (float): Center longitude used to determine UTM zone.
        center_y (float): Center latitude used to determine UTM zone.

    Returns:
        tuple[float, float, float, float]: Bounding box in UTM coordinates as (min_x, min_y, max_x, max_y).
    """
    zone_number = int((center_x + 180) / 6) + 1
    is_northern = center_y >= 0
    epsg_code = f"326{zone_number:02d}" if is_northern else f"327{zone_number:02d}"
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    utm_minx, utm_miny = transformer.transform(lon_min, lat_min)
    utm_maxx, utm_maxy = transformer.transform(lon_max, lat_max)
    return (utm_minx, utm_miny, utm_maxx, utm_maxy)


DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH
DEBUG = settings.DEBUG


@csrf_exempt
@api_view(["POST"])
@safe_api
def geocode(request) -> Response:
    """Resolve an address to latitude and longitude using Nominatim.

    Args:
        request (Request): POST request with an 'address' field.

    Returns:
        Response: JSON with 'lat' and 'lon' or error.
    """
    address = request.data.get("address")
    if not address:
        return Response({"error": "Address is required"}, status=400)
    coords = geocode_address(address)
    return Response({"lat": coords.lat, "lon": coords.lon})


def index(request) -> Response:
    """Render the main frontend index page.

    Args:
        request (Request): Django request.

    Returns:
        Response: Rendered HTML response.
    """
    return render(request, "core/index.html")


@csrf_exempt
@api_view(["POST"])
@safe_api
def slice_contours(request) -> Response:
    """Slice elevation data into contour layers for laser cutting.

    Args:
        request (Request): JSON POST request with slicing parameters and bounding box.

    Returns:
        Response: JSON with contour data or error message.
    """
    height = float(request.data["height_per_layer"])
    layers = int(request.data["num_layers"])
    simplify = float(request.data["simplify"])
    bounds = request.data["bounds"]

    lat_min = float(bounds["lat_min"])
    lon_min = float(bounds["lon_min"])
    lat_max = float(bounds["lat_max"])
    lon_max = float(bounds["lon_max"])

    substrate_size = float(request.data["substrate_size"])
    layer_thickness = float(request.data["layer_thickness"])

    center_x = (lon_min + lon_max) / 2
    center_y = (lat_min + lat_max) / 2

    logger.info(
        f"Slicing bounds ({lat_min}, {lon_min}) to ({lat_max}, {lon_max}) "
        f"with {height}m per layer, {layers} layers, simplify={simplify}"
    )

    # Download tiles
    tile_paths = download_srtm_tiles_for_bounds((lon_min, lat_min, lon_max, lat_max))

    logger.debug(f"downloaded paths: {tile_paths}")

    # Merge and clip to viewport
    elevation, transform = mosaic_and_crop(
        tile_paths, (lon_min, lat_min, lon_max, lat_max)
    )
    logger.debug("Merged and clipped.")

    # Generate contours and save a preview

    logger.debug("Calling generate_contours...")

    contours = generate_contours(
        elevation,
        transform,
        height,
        simplify,
        DEBUG_IMAGE_PATH,
        center=(center_x, center_y),
        scale=100,
        bounds=(lon_min, lat_min, lon_max, lat_max),
    )
    logger.info(f"Generated {len(contours)} contour polygons.")

    # Project each contour geometry to UTM coordinates
    contours = project_geometry(contours, center_x, center_y)

    utm_bounds = compute_utm_bounds_from_wgs84(
        lon_min, lat_min, lon_max, lat_max, center_x, center_y
    )
    logger.debug(
        f"UTM bounds: ({utm_bounds[0]:.2f}, {utm_bounds[1]:.2f}) → ({utm_bounds[2]:.2f}, {utm_bounds[3]:.2f}) "
        f"→ extent: {utm_bounds[2] - utm_bounds[0]:.2f} m × {utm_bounds[3] - utm_bounds[1]:.2f} m"
    )
    contours = scale_and_center_contours_to_substrate(
        contours, substrate_size, utm_bounds
    )
    for contour in contours:
        contour["thickness"] = layer_thickness / 1000.0  # in meters

    return Response(
        {"status": "sliced", "preview": DEBUG_IMAGE_PATH, "layers": contours}
    )
