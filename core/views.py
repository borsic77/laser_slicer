import io
import logging
from functools import wraps

from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response

from core.services.contour_generator import ContourSlicingJob
from core.services.elevation_service import ElevationDataError, ElevationRangeJob
from core.services.svg_zip_generator import generate_svg_layers, zip_svgs
from core.utils.export_filename import build_export_basename
from core.utils.geocoding import geocode_address

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
    bounds = _parse_bounds(request.data["bounds"])
    try:
        result = ElevationRangeJob(bounds).run()
        return Response(result)
    except ElevationDataError as e:
        logger.warning(f"Elevation data error for bounds {bounds}: {e}")
        return Response({"error": str(e)}, status=400)


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

    svg_files = generate_svg_layers(contours, basename=base_filename)
    zip_bytes = zip_svgs(svg_files)  # generate SVGs in memory

    response = FileResponse(
        io.BytesIO(zip_bytes),
        content_type="application/zip",
        as_attachment=True,
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    response["Access-Control-Expose-Headers"] = "Content-Disposition"
    return response


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


def _parse_bounds(bounds: dict) -> tuple[float, float, float, float]:
    """Parse bounding box coordinates from a dictionary.
    Args:
        bounds (dict): Dictionary containing 'lon_min', 'lat_min', 'lon_max', 'lat_max'.
    Returns:
        tuple[float, float, float, float]: Tuple of (lon_min, lat_min, lon_max, lat_max).
    """
    return (
        float(bounds["lon_min"]),
        float(bounds["lat_min"]),
        float(bounds["lon_max"]),
        float(bounds["lat_max"]),
    )


def _compute_center(bounds: dict) -> tuple[float, float]:
    """Compute the center of a bounding box.
    Args:
        bounds (dict): Dictionary containing 'lon_min', 'lat_min', 'lon_max', 'lat_max'.
    Returns:
        tuple[float, float]: Tuple of (lon_center, lat_center).
    """
    lat_min = float(bounds["lat_min"])
    lat_max = float(bounds["lat_max"])
    lon_min = float(bounds["lon_min"])
    lon_max = float(bounds["lon_max"])
    return ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)


@csrf_exempt
@api_view(["POST"])
@safe_api
def slice_contours(request):
    job = ContourSlicingJob(
        bounds=_parse_bounds(request.data["bounds"]),
        height_per_layer=float(request.data["height_per_layer"]),
        num_layers=int(request.data["num_layers"]),
        simplify=float(request.data["simplify"]),
        substrate_size_mm=float(request.data["substrate_size"]),
        layer_thickness_mm=float(request.data["layer_thickness"]),
        center=_compute_center(request.data["bounds"]),
    )
    layers = job.run()
    return Response(
        {"status": "sliced", "preview": settings.DEBUG_IMAGE_PATH, "layers": layers}
    )
