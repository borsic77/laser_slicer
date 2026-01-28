import uuid

from django.db import models


class BaseJob(models.Model):
    """Abstract base for all asynchronous jobs (contours, elevation, SVG, etc.).

    attributes:
        id (UUID): Unique identifier for the job.
        created_at (datetime): Timestamp when the job was created.
        started_at (datetime): Timestamp when the job processing started.
        finished_at (datetime): Timestamp when the job processing finished.
        status (str): Current status of the job (e.g., "PENDING", "RUNNING", "SUCCESS", "FAILURE").
        progress (int): Progress percentage (0-100).
        params (dict): JSON dictionary of job parameters.
        result_file (File): Link to the generated result file (e.g., ZIP archive).
        log (str): Internal log messages for the job.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20, default="PENDING"
    )  # E.g., PENDING, RUNNING, SUCCESS, FAILURE
    progress = models.PositiveSmallIntegerField(default=0)  # 0-100
    params = models.JSONField(default=dict, blank=True)
    result_file = models.FileField(null=True, blank=True, upload_to="media/")
    log = models.TextField(blank=True)

    class Meta:
        abstract = True

    @property
    def job_type(self):
        return self.__class__.__name__


class ContourJob(BaseJob):
    """Job for generating contour layers from an area."""

    pass


class ElevationJob(BaseJob):
    """Job for fetching elevation data without slicing."""

    pass


class SVGJob(BaseJob):
    """Job for exporting existing contours to SVG/ZIP."""

    pass
