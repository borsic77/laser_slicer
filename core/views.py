import io
import logging
import zipfile

from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from core.utils.geocoding import geocode_address
from core.utils.slicer import (
    download_srtm_tiles_for_bounds,
    generate_contours,
    mosaic_and_crop,
)

DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH

logger = logging.getLogger(__name__)


@api_view(["POST"])
def geocode(request):
    address = request.data.get("address")
    if not address:
        return Response({"error": "Address is required"}, status=400)
    try:
        lat, lon = geocode_address(address)
        return Response({"lat": lat, "lon": lon})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


def index(request):
    return render(request, "core/index.html")


@api_view(["POST"])
def slice_contours(request):
    try:
        height = float(request.data["height_per_layer"])
        layers = int(request.data["num_layers"])
        simplify = float(request.data["simplify"])
        bounds = request.data["bounds"]

        lat_min = float(bounds["lat_min"])
        lon_min = float(bounds["lon_min"])
        lat_max = float(bounds["lat_max"])
        lon_max = float(bounds["lon_max"])

        center_x = (lon_min + lon_max) / 2
        center_y = (lat_min + lat_max) / 2

        logger.info(
            f"Slicing bounds ({lat_min}, {lon_min}) to ({lat_max}, {lon_max}) "
            f"with {height}m per layer, {layers} layers, simplify={simplify}"
        )

        # Download tiles
        tile_paths = download_srtm_tiles_for_bounds(
            (lon_min, lat_min, lon_max, lat_max)
        )

        logger.debug(f"downloaded paths: {tile_paths}")

        # Merge and clip to viewport
        elevation, transform = mosaic_and_crop(
            tile_paths, (lon_min, lat_min, lon_max, lat_max)
        )
        logger.debug("Merged and clipped.")

        # Generate contours and save a preview

        try:
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
        except Exception as contour_error:
            logger.exception("Contour generation failed")
            raise

        return Response(
            {"status": "sliced", "preview": DEBUG_IMAGE_PATH, "layers": contours}
        )
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def export_svgs(request):
    # Create a dummy ZIP file with fake SVG content
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("layer_01.svg", "<svg><rect width='100' height='100'/></svg>")
        zf.writestr("layer_02.svg", "<svg><circle cx='50' cy='50' r='40'/></svg>")
    mem_zip.seek(0)
    return FileResponse(mem_zip, as_attachment=True, filename="contours.zip")
