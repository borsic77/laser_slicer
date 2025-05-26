import uuid

from django.db import models


class BaseJob(models.Model):
    """
    Abstract base for all asynchronous jobs (contours, elevation, SVG, etc.)
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
    pass


class ElevationJob(BaseJob):
    pass


class SVGJob(BaseJob):
    pass
