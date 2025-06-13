from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/geocode/", views.geocode),
    path("api/slice/", views.slice_contours),
    path("api/elevation-range/", views.elevation_range),
    path("api/export/", views.export_svgs_job),
    path("api/jobs/<uuid:job_id>/", views.job_status, name="job_status"),
    path("api/elevation", views.get_elevation),
    path("api/water-info/", views.water_info),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
