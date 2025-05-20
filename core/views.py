import io
import logging
import traceback
import zipfile

import numpy as np
from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render
from pyproj import Transformer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from shapely.geometry import mapping, shape

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


@api_view(["POST"])
def elevation_range(request):
    try:
        bounds = request.data["bounds"]

        lat_min = float(bounds["lat_min"])
        lon_min = float(bounds["lon_min"])
        lat_max = float(bounds["lat_max"])
        lon_max = float(bounds["lon_max"])

        tile_paths = download_srtm_tiles_for_bounds(
            (lon_min, lat_min, lon_max, lat_max)
        )
        elevation, _ = mosaic_and_crop(tile_paths, (lon_min, lat_min, lon_max, lat_max))

        masked = np.ma.masked_where(
            ~np.isfinite(elevation) | (elevation <= -32768), elevation
        )
        min_elev = max(-500, float(masked.min()))
        max_elev = min(10000, float(masked.max()))

        return Response({"min": min_elev, "max": max_elev})
    except Exception as e:
        logger.exception("Error getting elevation range")
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def export_svgs(request):
    contours = request.data.get("layers")  # front-end sends the list it already has
    if not contours:
        return Response({"error": "No layers supplied"}, status=400)

    try:
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
    except Exception as exc:
        logger.exception("SVG export failed")
        return Response({"error": str(exc)}, status=500)


def compute_utm_bounds_from_wgs84(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    center_x: float,
    center_y: float,
) -> tuple[float, float, float, float]:
    zone_number = int((center_x + 180) / 6) + 1
    is_northern = center_y >= 0
    epsg_code = f"326{zone_number:02d}" if is_northern else f"327{zone_number:02d}"
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    utm_minx, utm_miny = transformer.transform(lon_min, lat_min)
    utm_maxx, utm_maxy = transformer.transform(lon_max, lat_max)
    return (utm_minx, utm_miny, utm_maxx, utm_maxy)


DEBUG_IMAGE_PATH = settings.DEBUG_IMAGE_PATH

logger = logging.getLogger(__name__)


@api_view(["POST"])
def geocode(request):
    address = request.data.get("address")
    if not address:
        return Response({"error": "Address is required"}, status=400)
    try:
        coords = geocode_address(address)
        return Response({"lat": coords.lat, "lon": coords.lon})
    except Exception as e:
        logger.exception("Error geocoding address")
        traceback.print_exc()
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

        substrate_size = float(request.data["substrate_size"])
        layer_thickness = float(request.data["layer_thickness"])

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

        logger.debug("Calling generate_contours...")
        try:
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
        except Exception as e:
            logger.exception("Error generating contours")
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)
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
    except Exception as e:
        logger.exception("Error generating contours")
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
