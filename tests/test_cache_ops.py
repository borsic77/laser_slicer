"""Tests for cache size bounds and LRU eviction operations."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

from filelock import Timeout

from core.utils.cache_ops import (
    enforce_all_cache_limits,
    enforce_cache_limit,
    touch_cache_file,
)


def test_touch_cache_file_updates_mtime(tmp_path: Path):
    """Test that touch_cache_file updates the st_mtime of an existing file."""
    test_file = tmp_path / "tile.tif"
    test_file.write_bytes(b"test data")

    # Set timestamp in the past
    past_time = time.time() - 3600
    os.utime(test_file, (past_time, past_time))
    assert abs(test_file.stat().st_mtime - past_time) < 2

    # Touch cache file and verify mtime updated close to current time
    touch_cache_file(test_file)
    assert abs(test_file.stat().st_mtime - time.time()) < 5

    # Touching a nonexistent file should not raise an exception
    touch_cache_file(tmp_path / "nonexistent.tif")


def test_enforce_cache_limit_under_limit(tmp_path: Path):
    """Test that enforce_cache_limit does nothing when cache size <= max_bytes."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    f1 = cache_dir / "tile1.tif"
    f2 = cache_dir / "tile2.tif"
    f1.write_bytes(b"a" * 200)
    f2.write_bytes(b"b" * 300)

    freed = enforce_cache_limit(cache_dir, max_bytes=1000)
    assert freed == 0
    assert f1.exists()
    assert f2.exists()


def test_enforce_cache_limit_invalid_inputs(tmp_path: Path):
    """Test edge cases with non-existent directories or invalid max_bytes."""
    assert enforce_cache_limit(tmp_path / "does_not_exist", max_bytes=1000) == 0
    assert enforce_cache_limit(tmp_path, max_bytes=0) == 0
    assert enforce_cache_limit(tmp_path, max_bytes=-500) == 0


def test_enforce_cache_limit_evicts_oldest_and_cleans_companions(tmp_path: Path):
    """Test eviction of oldest files down to target_ratio, ignoring .lock/.tmp

    and unlinking companion .lock/.properties files.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    now = time.time()

    # Oldest file + companion lock and properties
    f1 = cache_dir / "tile1.tif"
    f1.write_bytes(b"1" * 300)
    os.utime(f1, (now - 300, now - 300))
    f1_lock = cache_dir / "tile1.tif.lock"
    f1_lock.write_bytes(b"lock1")
    f1_props = cache_dir / "tile1.tif.properties"
    f1_props.write_bytes(b"compressed_size=300")
    os.utime(f1_props, (now - 300, now - 300))

    # Middle file + companion files
    f2 = cache_dir / "tile2.tif"
    f2.write_bytes(b"2" * 300)
    os.utime(f2, (now - 200, now - 200))
    f2_lock = cache_dir / "tile2.tif.lock"
    f2_lock.write_bytes(b"lock2")
    f2_props = cache_dir / "tile2.tif.properties"
    f2_props.write_bytes(b"compressed_size=300")
    os.utime(f2_props, (now - 200, now - 200))

    # Newest file + companion files
    f3 = cache_dir / "tile3.tif"
    f3.write_bytes(b"3" * 300)
    os.utime(f3, (now - 100, now - 100))
    f3_lock = cache_dir / "tile3.tif.lock"
    f3_lock.write_bytes(b"lock3")
    f3_props = cache_dir / "tile3.tif.properties"
    f3_props.write_bytes(b"compressed_size=300")
    os.utime(f3_props, (now - 100, now - 100))

    # Files that must be ignored during eviction collection
    unrelated_lock = cache_dir / "unrelated.lock"
    unrelated_lock.write_bytes(b"unrelated lock")
    tmp_file = cache_dir / "tile4.tmp.gz"
    tmp_file.write_bytes(b"temp download in progress")

    # Set max_bytes=700, target_ratio=0.5 -> target_bytes=350.
    # Total candidates exceed 700 bytes, so eviction occurs down to <= 350 bytes.
    freed = enforce_cache_limit(cache_dir, max_bytes=700, target_ratio=0.5)

    assert freed > 0
    # Oldest (f1) must be evicted along with companion files
    assert not f1.exists()
    assert not f1_lock.exists()
    assert not f1_props.exists()

    # Second oldest (f2) must be evicted to reach <= 350 bytes
    assert not f2.exists()
    assert not f2_lock.exists()
    assert not f2_props.exists()

    # Newest file (f3) and companion files must remain
    assert f3.exists()
    assert f3_lock.exists()
    assert f3_props.exists()

    # Unrelated lock and temporary download file must not be evicted
    assert unrelated_lock.exists()
    assert tmp_file.exists()


def test_enforce_cache_limit_lock_failure(tmp_path: Path, monkeypatch):
    """Test graceful handling when eviction lock acquisition times out."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "test.tif").write_bytes(b"data" * 100)

    mock_lock = MagicMock()
    mock_lock.__enter__.side_effect = Timeout("Lock acquisition timed out")
    monkeypatch.setattr(
        "core.utils.cache_ops.FileLock", lambda *args, **kwargs: mock_lock
    )

    freed = enforce_cache_limit(cache_dir, max_bytes=10)
    assert freed == 0


def test_enforce_all_cache_limits_with_settings(tmp_path: Path, settings):
    """Test enforce_all_cache_limits with configured cache directories and limits."""
    tile_dir = tmp_path / "tile_cache"
    alti3d_dir = tmp_path / "alti3d_cache"
    bathy_dir = tmp_path / "bathy_cache"
    osm_dir = tmp_path / "osm_cache"
    dem_dir = tmp_path / "processed_dem_cache"

    for d in (tile_dir, alti3d_dir, bathy_dir, osm_dir, dem_dir):
        d.mkdir()

    settings.TILE_CACHE_DIR = tile_dir
    settings.ALTI3D_CACHE_DIR = alti3d_dir
    settings.BATHY_CACHE_DIR = bathy_dir
    settings.OSM_CACHE_DIR = osm_dir
    settings.PROCESSED_DEM_CACHE_DIR = dem_dir

    # Limit to 1 MB each
    settings.TILE_CACHE_MAX_MB = 1
    settings.ALTI3D_CACHE_MAX_MB = 1
    settings.BATHY_CACHE_MAX_MB = 1
    settings.OSM_CACHE_MAX_MB = 1
    settings.PROCESSED_DEM_CACHE_MAX_MB = 1

    # Overfill TILE_CACHE_DIR with 1.6 MB (two 800 KB files)
    t1 = tile_dir / "tile1.tif"
    t2 = tile_dir / "tile2.tif"
    t1.write_bytes(b"x" * (800 * 1024))
    t2.write_bytes(b"y" * (800 * 1024))
    now = time.time()
    os.utime(t1, (now - 200, now - 200))
    os.utime(t2, (now - 100, now - 100))

    # Keep BATHY_CACHE_DIR under limit (100 KB)
    (bathy_dir / "bathy.tif").write_bytes(b"z" * (100 * 1024))

    results = enforce_all_cache_limits()

    expected_keys = {
        "TILE_CACHE_DIR",
        "ALTI3D_CACHE_DIR",
        "BATHY_CACHE_DIR",
        "OSM_CACHE_DIR",
        "PROCESSED_DEM_CACHE_DIR",
    }
    assert set(results.keys()) == expected_keys
    assert results["TILE_CACHE_DIR"] >= 800 * 1024
    assert not t1.exists()  # Oldest file evicted
    assert t2.exists()  # Newest file retained
    assert results["BATHY_CACHE_DIR"] == 0
    assert results["ALTI3D_CACHE_DIR"] == 0
    assert results["OSM_CACHE_DIR"] == 0
    assert results["PROCESSED_DEM_CACHE_DIR"] == 0
