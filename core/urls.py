from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/geocode/", views.geocode),
    path("api/slice/", views.slice_contours),
    path("api/export_svg/", views.export_svgs),
]
