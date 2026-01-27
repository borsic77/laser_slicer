import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.utils.download_clip_elevation_tiles import (
    SWISS_BOUNDS,
    download_alti3d_tiles_for_bounds,
    within_swiss_bounds,
)


def test_within_swiss_bounds():
    # Inside Switzerland
    assert within_swiss_bounds((7.0, 46.0, 8.0, 47.0)) is True
    # Overlapping
    assert within_swiss_bounds((5.0, 45.0, 11.0, 48.0)) is True
    # Outside
    assert within_swiss_bounds((0.0, 0.0, 1.0, 1.0)) is False
    assert within_swiss_bounds((-10.0, 40.0, -9.0, 41.0)) is False


@patch("shutil.which")
def test_download_alti3d_missing_binary(mock_which):
    mock_which.return_value = None
    with pytest.raises(RuntimeError, match="Neither '.*' nor 'docker' were found"):
        download_alti3d_tiles_for_bounds(SWISS_BOUNDS)


@patch("shutil.which")
@patch("subprocess.run")
def test_download_alti3d_success(mock_run, mock_which, tmp_path):
    mock_which.return_value = "/usr/bin/alti3d-downloader"
    mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

    # We need to mock the CACHE_DIR provided by settings, but since it's imported at module level,
    # we might need to patch the reference in the module or use the tmp_path if checking file outputs.
    # For now, we trust the function returns what glob finds.

    with patch("core.utils.download_clip_elevation_tiles.ALTI3D_CACHE_DIR", tmp_path):
        # Create a dummy file to be found
        (tmp_path / "test.tif").touch()

        results = download_alti3d_tiles_for_bounds((6.0, 46.0, 6.1, 46.1))

        assert len(results) == 1
        assert "test.tif" in results[0]
        assert mock_run.called


@patch("shutil.which")
@patch("subprocess.run")
def test_download_alti3d_failure(mock_run, mock_which):
    mock_which.return_value = "/usr/bin/alti3d-downloader"
    mock_run.side_effect = subprocess.CalledProcessError(1, ["cmd"])

    with pytest.raises(RuntimeError, match="Downloader failed"):
        download_alti3d_tiles_for_bounds(SWISS_BOUNDS)


@patch("shutil.which")
@patch("subprocess.run")
@patch("core.utils.download_clip_elevation_tiles.ALTI3D_CACHE_DIR")
def test_download_alti3d_docker_fallback(mock_cache, mock_run, mock_which):
    # Simulate binary missing but docker present
    mock_which.side_effect = lambda x: "/usr/bin/docker" if x == "docker" else None

    mock_cache.resolve.return_value = Path("/abs/path/to/cache")
    mock_cache.glob.return_value = []  # Return empty list just for flow

    download_alti3d_tiles_for_bounds(SWISS_BOUNDS)

    assert mock_run.called
    args = mock_run.call_args[0][0]
    assert args[0] == "docker"
    assert "local/alti3d-downloader" in args
