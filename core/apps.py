from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        import osmnx as ox
        from django.conf import settings

        ox.settings.use_cache = True
        ox.settings.cache_folder = settings.OSM_CACHE_DIR
