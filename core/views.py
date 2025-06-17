import logging
from functools import wraps

from django.apps import apps
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from rest_framework import status as drf_status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from shapely.geometry import mapping

from core.models import BaseJob, ContourJob, ElevationJob, SVGJob
from core.tasks import (  # New: see below
    run_contour_slicing_job,
    run_elevation_range_job,
    run_svg_export_job,
)
from core.utils.download_clip_elevation_tiles import ensure_tile_downloaded
from core.utils.geocoding import geocode_address
from core.utils.slicer import sample_elevation
from core.utils.waterbody import fetch_waterbody_polygon

logger = logging.getLogger(__name__)


def safe_api(view_func):
    """Decorator to handle exceptions in API views.
    Logs the exception and returns a generic error response.
    Args:
        view_func (callable): The view function to wrap.
    Returns:
        callable: The wrapped view function.
    """

    @wraps(view_func)
    def wrapper(*args, **kwargs):
        try:
            return view_func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"API failure in {view_func.__name__}")
            return Response({"error": "An unexpected error occurred."}, status=500)

    return wrapper


@require_GET
def get_elevation(request):
    """Fetch elevation data for a specific latitude and longitude.
    Args:
        request (HttpRequest): GET request with 'lat' and 'lon' parameters.
    Returns:
        JsonResponse: JSON with elevation data or error message.
    """
    logger.debug("Received elevation request with params: %s", request.GET)
    try:
        lat = float(request.GET["lat"])
        lon = float(request.GET["lon"])
    except (KeyError, ValueError):
        return JsonResponse(
            {"detail": "Invalid or missing lat/lon", "code": "bad_request"},
            status=400,
        )

    # Use your slicer/dem logic: you may need to determine which DEM to open
    dem_path = ensure_tile_downloaded(lat, lon)
    if dem_path is None:
        return JsonResponse(
            {"detail": "DEM not found for this location", "code": "not_found"},
            status=404,
        )
    try:
        elevation = sample_elevation(lat, lon, dem_path)
    except Exception as e:
        return JsonResponse(
            {"detail": f"Could not retrieve elevation: {e}", "code": "dem_error"},
            status=500,
        )
    logger.debug("Elevation for (%s, %s): %s", lat, lon, elevation)
    return JsonResponse({"elevation": round(elevation, 1)})


@require_GET
def waterbody(request):
    """Return the waterbody polygon containing a point if any."""
    try:
        lat = float(request.GET["lat"])
        lon = float(request.GET["lon"])
    except (KeyError, ValueError):
        return JsonResponse({"detail": "Invalid or missing lat/lon"}, status=400)

    polygon = fetch_waterbody_polygon(lat, lon)
    if polygon is not None:
        return JsonResponse({"in_water": True, "polygon": mapping(polygon)})
    return JsonResponse({"in_water": False, "polygon": None})


@csrf_exempt
@api_view(["POST"])
@safe_api
def elevation_range(request) -> Response:
    """
    Create a job to fetch elevation data for a bounding box.
    Args:
        request (Request): POST request with bounding box coordinates.
    Returns:
        Response: JSON with job ID or error message.
    """
    logger.debug("Received elevation range request with data: %s", request.data)
    params = {
        "bounds": request.data["bounds"],
    }
    job = ElevationJob.objects.create(params=params, status="PENDING")
    run_elevation_range_job.delay(str(job.id))
    return Response({"job_id": str(job.id)}, status=202)


@csrf_exempt
@api_view(["POST"])
@safe_api
def export_svgs_job(request):
    params = {
        "layers": request.data.get("layers"),
        "address": request.data.get("address", "").strip(),
        "coordinates": request.data.get("coordinates", None),
        "height_per_layer": request.data.get("height_per_layer", "unknown"),
    }
    job = SVGJob.objects.create(params=params, status="PENDING")
    run_svg_export_job.delay(str(job.id))
    return Response({"job_id": str(job.id)}, status=202)


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
    """Create a contour slicing job with the provided parameters.
    Args:
        request (Request): POST request with slicing parameters.
    Returns:
        Response: JSON with job ID or error message.
    """
    fixed_elevation_value = request.data.get("fixedElevation", None)
    params = {
        "bounds": request.data["bounds"],
        "height_per_layer": float(request.data["height_per_layer"]),
        "num_layers": int(request.data["num_layers"]),
        "simplify": float(request.data["simplify"]),
        "substrate_size_mm": float(request.data["substrate_size"]),
        "layer_thickness_mm": float(request.data["layer_thickness"]),
        "center": _compute_center(request.data["bounds"]),
        "smoothing": int(request.data.get("smoothing", 0)),
        "min_area": float(request.data.get("min_area", 0.0)),
        "min_feature_width_mm": float(request.data.get("min_feature_width", 0.0)),
        "fixed_elevation": float(fixed_elevation_value)
        if fixed_elevation_value not in (None, "", "null")
        else None,
        "water_polygon": request.data.get("water_polygon"),
    }
    # logger.debug("Creating contour slicing job with params: %s", params)
    job = ContourJob.objects.create(params=params, status="PENDING")
    # Start async task
    run_contour_slicing_job.delay(str(job.id))
    return Response({"job_id": str(job.id)}, status=202)


def get_all_job_models():
    """Return all non-abstract models inheriting from BaseJob."""
    job_models = []
    for model in apps.get_models():
        if issubclass(model, BaseJob) and not model._meta.abstract:
            job_models.append(model)
    return job_models


@api_view(["GET"])
@safe_api
def job_status(request, job_id):
    job = None
    for model in get_all_job_models():
        try:
            job = model.objects.get(pk=job_id)
            break
        except model.DoesNotExist:
            continue
    if not job:
        return Response({"error": "Job not found"}, status=404)
    data = {
        "status": job.status,
        "progress": getattr(job, "progress", None),
        "log": getattr(job, "log", "")[-500:],  # last N chars
        "created_at": job.created_at,
        "started_at": getattr(job, "started_at", None),
        "finished_at": getattr(job, "finished_at", None),
        "result_url": job.result_file.url
        if getattr(job, "result_file", None)
        else None,
        "params": getattr(job, "params", None),
    }
    return Response(data, status=drf_status.HTTP_200_OK)
