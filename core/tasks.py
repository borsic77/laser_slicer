import logging
import time
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

from core.models import ContourJob, ElevationJob, SVGJob
from core.services.contour_generator import ContourSlicingJob
from core.services.elevation_service import ElevationDataError, ElevationRangeJob
from core.services.svg_zip_generator import generate_svg_layers, zip_svgs
from core.utils.cache_ops import enforce_all_cache_limits
from core.utils.export_filename import build_export_basename
from core.utils.geocoding import _parse_bounds

logger = logging.getLogger(__name__)


@shared_task
def cleanup_old_media(hours=1):
    """Clean up media files and expired job records older than `hours`."""
    cutoff_ts = time.time() - hours * 3600
    media_dir = Path(settings.MEDIA_ROOT)
    if media_dir.exists():
        for file in media_dir.glob("**/*"):
            try:
                if file.is_file() and file.stat().st_mtime < cutoff_ts:
                    file.unlink(missing_ok=True)
            except OSError:
                pass

    cutoff_dt = timezone.now() - timezone.timedelta(hours=hours)
    for model in (ContourJob, ElevationJob, SVGJob):
        model.objects.filter(created_at__lt=cutoff_dt).delete()

    enforce_all_cache_limits()


@shared_task(bind=True)
def run_svg_export_job(self, job_id):
    """
    Task to run the SVG export job.
    This generates SVG files from the contours provided in the job parameters,
    zips them, and saves the result to the job's result_file.
    Args:
        self: The Celery task instance.
        job_id (str): The ID of the SVGJob to run.
    """
    job = SVGJob.objects.get(pk=job_id)
    try:
        job.status = "RUNNING"
        job.started_at = timezone.now()
        job.save(update_fields=["status", "started_at"])
        params = job.params
        contours = params.get("layers")
        if not contours:
            raise ValueError("No layers supplied")
        address = params.get("address", "").strip()
        coords = params.get("coordinates", None)
        height_mm = params.get("height_per_layer", "unknown")
        num_layers = len(contours)
        base_filename = build_export_basename(address, coords, height_mm, num_layers)
        filename = f"{base_filename}.zip"

        svg_files = generate_svg_layers(contours, basename=base_filename)
        zip_bytes = zip_svgs(svg_files)

        # Save file without committing model until status/fields are updated
        job.result_file.save(filename, ContentFile(zip_bytes), save=False)
        job.status = "SUCCESS"
        job.finished_at = timezone.now()
        job.progress = 100
        job.log += "\nSVGs exported and zipped."
        job.save(
            update_fields=["result_file", "status", "finished_at", "progress", "log"]
        )
    except Exception as exc:
        logger.exception("SVG export job %s failed", job_id)
        job.status = "FAILURE"
        job.finished_at = timezone.now()
        job.log += f"\nExport failed: {exc}"
        job.save(update_fields=["status", "finished_at", "log"])
        raise


@shared_task(bind=True)
def run_elevation_range_job(self, job_id):
    """
    Task to run the elevation range job.
    This fetches elevation data for a given bounding box and updates the job status.
    Args:
        self: The Celery task instance.
        job_id (str): The ID of the ElevationJob to run.
    """
    job = ElevationJob.objects.get(pk=job_id)
    try:
        job.status = "RUNNING"
        job.started_at = timezone.now()
        job.save(update_fields=["status", "started_at"])
        params = job.params
        bounds_dict = params["bounds"]
        # Parse bounds from the dictionary
        bounds = (
            float(bounds_dict["lon_min"]),
            float(bounds_dict["lat_min"]),
            float(bounds_dict["lon_max"]),
            float(bounds_dict["lat_max"]),
        )
        include_bathymetry = params.get("include_bathymetry", False)
        result = ElevationRangeJob(bounds, include_bathymetry=include_bathymetry).run()
        job.params["result"] = result
        job.status = "SUCCESS"
        job.finished_at = timezone.now()
        job.progress = 100
        job.log += "\nElevation data fetched successfully."
        job.result_file = None
        job.save(
            update_fields=[
                "params",
                "status",
                "finished_at",
                "progress",
                "log",
                "result_file",
            ]
        )
        logger.info(f"Elevation job {job_id} completed successfully. Result: {result}")
    except ElevationDataError as e:
        job.status = "FAILURE"
        job.finished_at = timezone.now()
        job.log += f"\nElevation data error: {e}"
        job.save(update_fields=["status", "finished_at", "log"])
    except Exception:
        logger.exception("Elevation range job %s failed", job_id)
        job.status = "FAILURE"
        job.finished_at = timezone.now()
        job.log += "\nFailed to calculate elevation range due to an internal error."
        job.save(update_fields=["status", "finished_at", "log"])
        raise


@shared_task(bind=True)
def run_contour_slicing_job(self, job_id):
    """
    Task to run the contour slicing job.
    This processes the contour job parameters, runs the slicing algorithm,
    and saves the resulting layers to the job parameters.
    Args:
        self: The Celery task instance.
        job_id (str): The ID of the ContourJob to run.
    """
    job = ContourJob.objects.get(pk=job_id)
    try:
        job.status = "RUNNING"
        job.started_at = timezone.now()
        job.save(update_fields=["status", "started_at"])
        params = job.params
        csj = ContourSlicingJob(
            bounds=_parse_bounds(params["bounds"]),
            height_per_layer=params["height_per_layer"],
            num_layers=params["num_layers"],
            simplify=params["simplify"],
            substrate_size_mm=params["substrate_size_mm"],
            layer_thickness_mm=params["layer_thickness_mm"],
            center=params["center"],
            smoothing=params["smoothing"],
            min_area=params["min_area"],
            min_feature_width_mm=params["min_feature_width_mm"],
            fixed_elevation=params["fixed_elevation"],
            water_polygon=params.get("water_polygon"),
            include_roads=params.get("include_roads", False),
            include_buildings=params.get("include_buildings", False),
            include_waterways=params.get("include_waterways", False),
            include_bathymetry=params.get("include_bathymetry", False),
        )

        # Main processing
        def update_progress(msg, pct):
            job.status = "RUNNING"
            job.progress = pct
            entry = f"[{pct}%] {msg}"
            job.log = f"{job.log}\n{entry}" if job.log else entry
            job.save(update_fields=["status", "progress", "log"])

        layers = csj.run(progress_callback=update_progress)
        job.params["layers"] = layers
        job.status = "SUCCESS"
        job.finished_at = timezone.now()
        job.progress = 100
        job.log += "\nDone."
        job.save(update_fields=["params", "status", "finished_at", "progress", "log"])
        logger.info(f"Contour job {job_id} completed with {len(layers)} layers.")
    except ElevationDataError as e:
        job.status = "FAILURE"
        job.finished_at = timezone.now()
        job.log += f"\nElevation data error: {e}"
        job.save(update_fields=["status", "finished_at", "log"])
    except Exception:
        logger.exception("Contour slicing job %s failed", job_id)
        job.status = "FAILURE"
        job.finished_at = timezone.now()
        job.log += "\nContour slicing failed due to an internal error."
        job.save(update_fields=["status", "finished_at", "log"])
        raise
