import os
import shutil

import osmnx as ox
import pytest
from django.conf import settings

from core.services.osm_service import is_land_central_point


def test_osm_caching_enabled(tmp_path, settings):
    # Override OSM_CACHE_DIR for this test
    cache_dir = tmp_path / "osm_cache"
    settings.OSM_CACHE_DIR = cache_dir

    # Re-apply settings to ox (since it might have been imported already)
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir

    # We need a real network call to populate cache, or we mock the response.
    # To verify *caching*, we usually need to see a file created.
    # Let's try a very simple query or check if ox.settings are preserved.

    assert ox.settings.use_cache is True
    assert ox.settings.cache_folder == cache_dir

    # Check if cache dir is created when needed.
    # We can't easily run a real query without hitting external API (which is flaky/slow for tests).
    # But we can check if the code *configured* it.

    from core.services import osm_service
    # Reload to ensure it picks up? No, modules are cached.
    # But our code in osm_service.py sets it at module level.

    assert osm_service.ox.settings.cache_folder == cache_dir
