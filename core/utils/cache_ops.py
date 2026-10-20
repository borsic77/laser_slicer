"""Disk cache management and bounded LRU eviction utilities."""

import logging
import os
from pathlib import Path

from django.conf import settings
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

CACHE_CONFIGS = [
    ("TILE_CACHE_DIR", "TILE_CACHE_MAX_MB", 1024),
    ("ALTI3D_CACHE_DIR", "ALTI3D_CACHE_MAX_MB", 1024),
    ("BATHY_CACHE_DIR", "BATHY_CACHE_MAX_MB", 512),
    ("OSM_CACHE_DIR", "OSM_CACHE_MAX_MB", 512),
    ("PROCESSED_DEM_CACHE_DIR", "PROCESSED_DEM_CACHE_MAX_MB", 512),
]


def touch_cache_file(path: Path | str) -> None:
    """Update file modification time to current time for LRU tracking.

    Args:
        path: Path to the cache file.
    """
    try:
        os.utime(path, None)
    except OSError:
        pass


def enforce_cache_limit(
    cache_dir: Path | str,
    max_bytes: int,
    target_ratio: float = 0.85,
) -> int:
    """Enforce maximum disk cache size by evicting oldest files.

    Args:
        cache_dir: Path to the cache directory.
        max_bytes: Maximum allowed cache size in bytes. If <= 0, no eviction.
        target_ratio: Target ratio of max_bytes to retain after eviction (default 0.85).

    Returns:
        Total number of bytes freed.
    """
    if max_bytes <= 0:
        return 0

    cache_path = Path(cache_dir)
    if not cache_path.exists() or not cache_path.is_dir():
        return 0

    lock_path = cache_path / ".evict.lock"
    try:
        with FileLock(str(lock_path), timeout=2):
            candidates: list[tuple[Path, int, float]] = []
            candidate_sizes: dict[Path, int] = {}
            total_bytes = 0

            for p in cache_path.rglob("*"):
                if not p.is_file():
                    continue
                if p.name.endswith(".lock") or ".tmp" in p.name:
                    continue
                try:
                    st = p.stat()
                    candidates.append((p, st.st_size, st.st_mtime))
                    candidate_sizes[p] = st.st_size
                    total_bytes += st.st_size
                except OSError:
                    continue

            if total_bytes <= max_bytes:
                return 0

            target_bytes = int(max_bytes * target_ratio)
            # Sort oldest first; if timestamps match, ensure .properties is sorted after base file
            candidates.sort(
                key=lambda item: (
                    item[2],
                    1 if item[0].name.endswith(".properties") else 0,
                )
            )

            bytes_freed = 0
            files_evicted = 0
            deleted_paths: set[Path] = set()

            for file_path, size, _ in candidates:
                if total_bytes <= target_bytes:
                    break
                if file_path in deleted_paths or not file_path.exists():
                    continue

                try:
                    file_path.unlink(missing_ok=True)
                    deleted_paths.add(file_path)
                    total_bytes -= size
                    bytes_freed += size
                    files_evicted += 1
                except OSError as exc:
                    logger.warning("Failed to unlink %s: %s", file_path, exc)
                    continue

                # Remove companion .lock and .properties files
                for suffix in (".lock", ".properties"):
                    companion = Path(f"{file_path}{suffix}")
                    try:
                        if companion.is_file():
                            comp_size = candidate_sizes.get(companion, 0)
                            companion.unlink(missing_ok=True)
                            deleted_paths.add(companion)
                            if comp_size > 0:
                                total_bytes -= comp_size
                                bytes_freed += comp_size
                    except OSError:
                        pass

            if files_evicted > 0:
                logger.info(
                    "Evicted %d files (%.2f MB) from %s, freed %d bytes",
                    files_evicted,
                    bytes_freed / (1024 * 1024),
                    cache_path,
                    bytes_freed,
                )

            return bytes_freed

    except (Timeout, OSError) as exc:
        logger.warning("Could not acquire eviction lock on %s: %s", cache_path, exc)
        return 0
    except Exception as exc:
        logger.warning("Eviction lock on %s failed: %s", cache_path, exc)
        return 0


def enforce_all_cache_limits() -> dict[str, int]:
    """Enforce size limits across all configured disk caches.

    Returns:
        Dictionary mapping cache setting name to number of bytes freed.
    """
    results: dict[str, int] = {}
    for dir_setting, max_mb_setting, default_mb in CACHE_CONFIGS:
        cache_dir = getattr(settings, dir_setting, None)
        max_mb = getattr(settings, max_mb_setting, default_mb)
        if cache_dir is None:
            results[dir_setting] = 0
            continue
        max_bytes = int(max_mb) * 1024 * 1024
        freed = enforce_cache_limit(cache_dir, max_bytes)
        results[dir_setting] = freed
    return results
