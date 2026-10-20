import logging
import math
from functools import wraps

from django.apps import apps
from django.http import JsonResponse
from django.shortcuts import render
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
from core.utils.dem import ensure_tile_downloaded, sample_elevation
from core.utils.geocoding import geocode_address
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
        except Exception:
            logger.exception(f"API failure in {view_func.__name__}")
            return Response({"error": "An unexpected error occurred."}, status=500)

    return wrapper


def _validate_lat_lon(lat_raw, lon_raw) -> tuple[float, float]:
    lat = float(lat_raw)
    lon = float(lon_raw)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Coordinates must be finite numbers.")
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError("Coordinates out of range.")
    return lat, lon


def _validate_bounds_dict(bounds: dict, max_span_deg: float = 5.0) -> dict[str, float]:
    if not isinstance(bounds, dict):
        raise ValueError("bounds must be an object.")
    lon_min = float(bounds["lon_min"])
    lat_min = float(bounds["lat_min"])
    lon_max = float(bounds["lon_max"])
    lat_max = float(bounds["lat_max"])
    if not all(math.isfinite(v) for v in (lon_min, lat_min, lon_max, lat_max)):
        raise ValueError("bounds coordinates must be finite.")
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError("Invalid latitude bounds.")
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        raise ValueError("Invalid longitude bounds.")
    if (lat_max - lat_min) > max_span_deg or (lon_max - lon_min) > max_span_deg:
        raise ValueError(
            f"Bounding box exceeds maximum span of {max_span_deg} degrees."
        )
    return {
        "lon_min": lon_min,
        "lat_min": lat_min,
        "lon_max": lon_max,
        "lat_max": lat_max,
    }


@require_GET
def get_elevation(request):
    """Fetch elevation data for a specific latitude and longitude."""
    logger.debug("Received elevation request with params: %s", request.GET)
    try:
        lat, lon = _validate_lat_lon(request.GET["lat"], request.GET["lon"])
    except (KeyError, ValueError, TypeError):
        return JsonResponse(
            {"detail": "Invalid or missing lat/lon", "code": "bad_request"},
            status=400,
        )

    try:
        dem_path = ensure_tile_downloaded(lat, lon)
    except Exception:
        logger.exception("Failed to download DEM tile for (%s, %s)", lat, lon)
        return JsonResponse(
            {"detail": "Could not retrieve elevation tile", "code": "dem_error"},
            status=500,
        )

    if dem_path is None:
        return JsonResponse(
            {"detail": "DEM not found for this location", "code": "not_found"},
            status=404,
        )
    try:
        elevation = sample_elevation(lat, lon, str(dem_path))
    except Exception:
        logger.exception("Failed to sample elevation at (%s, %s)", lat, lon)
        return JsonResponse(
            {"detail": "Could not retrieve elevation", "code": "dem_error"},
            status=500,
        )
    if not math.isfinite(elevation) or elevation <= -32768:
        return JsonResponse(
            {"detail": "No valid elevation data at this point", "code": "not_found"},
            status=404,
        )
    logger.debug("Elevation for (%s, %s): %s", lat, lon, elevation)
    return JsonResponse({"elevation": round(elevation, 1)})


@require_GET
def waterbody(request):
    """Return the waterbody polygon containing a point if any."""
    try:
        lat, lon = _validate_lat_lon(request.GET["lat"], request.GET["lon"])
    except (KeyError, ValueError, TypeError):
        return JsonResponse({"detail": "Invalid or missing lat/lon"}, status=400)

    polygon = fetch_waterbody_polygon(lat, lon)
    if polygon is not None:
        return JsonResponse({"in_water": True, "polygon": mapping(polygon)})
    return JsonResponse({"in_water": False, "polygon": None})


@api_view(["POST"])
@safe_api
def elevation_range(request) -> Response:
    """Create a job to fetch elevation data for a bounding box."""
    logger.debug("Received elevation range request with data: %s", request.data)
    try:
        bounds = _validate_bounds_dict(request.data["bounds"], max_span_deg=5.0)
    except (KeyError, ValueError, TypeError) as exc:
        return Response({"error": f"Invalid bounds: {exc}"}, status=400)

    params = {
        "bounds": bounds,
        "include_bathymetry": bool(request.data.get("include_bathymetry", False)),
    }
    job = ElevationJob.objects.create(params=params, status="PENDING")
    run_elevation_range_job.delay(str(job.id))
    return Response({"job_id": str(job.id)}, status=202)


@api_view(["POST"])
@safe_api
def export_svgs_job(request):
    layers = request.data.get("layers")
    if not isinstance(layers, list) or not layers:
        return Response({"error": "A non-empty layers list is required"}, status=400)
    raw_address = request.data.get("address")
    params = {
        "layers": layers,
        "address": raw_address.strip() if isinstance(raw_address, str) else "",
        "coordinates": request.data.get("coordinates", None),
        "height_per_layer": request.data.get("height_per_layer", "unknown"),
    }
    job = SVGJob.objects.create(params=params, status="PENDING")
    run_svg_export_job.delay(str(job.id))
    return Response({"job_id": str(job.id)}, status=202)


@api_view(["POST"])
@safe_api
def geocode(request) -> Response:
    """Resolve an address to latitude and longitude using Nominatim."""
    address = request.data.get("address")
    if not address or not isinstance(address, str) or not address.strip():
        return Response({"error": "Address is required"}, status=400)
    try:
        coords = geocode_address(address.strip())
    except ValueError as exc:
        return Response({"error": str(exc)}, status=404)
    return Response({"lat": coords.lat, "lon": coords.lon})


def index(request):
    """Render the main frontend index page."""
    return render(request, "core/index.html")


def _compute_center(bounds: dict) -> tuple[float, float]:
    """Compute the center of a bounding box."""
    lat_min = float(bounds["lat_min"])
    lat_max = float(bounds["lat_max"])
    lon_min = float(bounds["lon_min"])
    lon_max = float(bounds["lon_max"])
    return ((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)


@api_view(["POST"])
@safe_api
def slice_contours(request):
    """Create a contour slicing job with the provided parameters."""
    try:
        bounds = _validate_bounds_dict(request.data["bounds"], max_span_deg=5.0)
        height_per_layer = float(request.data["height_per_layer"])
        num_layers = int(request.data["num_layers"])
        simplify = float(request.data["simplify"])
        substrate_size_mm = float(request.data["substrate_size"])
        layer_thickness_mm = float(request.data["layer_thickness"])
        smoothing = int(request.data.get("smoothing", 0))
        min_area = float(request.data.get("min_area", 0.0))
        min_feature_width_mm = float(request.data.get("min_feature_width", 0.0))
        fixed_elevation_value = request.data.get("fixedElevation", None)
        fixed_elevation = (
            float(fixed_elevation_value)
            if fixed_elevation_value not in (None, "", "null")
            else None
        )
    except (KeyError, ValueError, TypeError) as exc:
        return Response({"error": f"Invalid parameter: {exc}"}, status=400)

    if not (1 <= num_layers <= 200):
        return Response({"error": "num_layers must be between 1 and 200"}, status=400)
    if height_per_layer <= 0 or substrate_size_mm <= 0 or layer_thickness_mm <= 0:
        return Response(
            {
                "error": "height_per_layer, substrate_size, and layer_thickness must be positive"
            },
            status=400,
        )

    params = {
        "bounds": bounds,
        "height_per_layer": height_per_layer,
        "num_layers": num_layers,
        "simplify": max(0.0, simplify),
        "substrate_size_mm": substrate_size_mm,
        "layer_thickness_mm": layer_thickness_mm,
        "center": _compute_center(bounds),
        "smoothing": max(0, smoothing),
        "min_area": max(0.0, min_area),
        "min_feature_width_mm": max(0.0, min_feature_width_mm),
        "fixed_elevation": fixed_elevation,
        "water_polygon": request.data.get("water_polygon"),
        "include_roads": bool(request.data.get("include_roads", False)),
        "include_buildings": bool(request.data.get("include_buildings", False)),
        "include_waterways": bool(request.data.get("include_waterways", False)),
        "include_bathymetry": bool(request.data.get("include_bathymetry", False)),
    }
    job = ContourJob.objects.create(params=params, status="PENDING")
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
    """Retrieve the status of a specific job by its ID.

    Searches for the job ID across all job models (ContourJob, ElevationJob, SVGJob).

    Args:
        request (Request): The incoming request.
        job_id (str): UUID of the job to check.

    Returns:
        Response: JSON object containing job status, progress, logs, and result URL.
    """
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
