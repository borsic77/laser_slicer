from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from affine import Affine

from core.services.bathymetry_service import BathymetryFetcher


@patch("core.services.bathymetry_service.rasterio.open")
def test_fetch_elevation_for_bounds(mock_open, tmp_path):
    # Setup mock
    mock_src = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_src

    # Mock read data (e.g. 10x10 array)
    mock_data = np.zeros((10, 10)) - 100.0  # -100m depth
    mock_src.read.return_value = mock_data
    # Use real Affine object with negative Y scale (standard for GeoTIFF)
    mock_src.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 100.0)
    mock_src.nodata = None

    # Mock window transform
    mock_window_transform = MagicMock()
    mock_src.window_transform.return_value = mock_window_transform

    # Instantiate service
    service = BathymetryFetcher(cache_dir=tmp_path)

    # Call
    bounds = (10, 10, 11, 11)
    result = service.fetch_elevation_for_bounds(bounds)

    # Verify
    assert result.shape == (10, 10)
    assert np.all(result == -100.0)
    mock_src.read.assert_called()

    # Check cache file creation
    # The service writes to cache, so we expect a file in tmp_path
    # Filename format: etopo_{lon_min:.4f}_{lat_min:.4f}_{lon_max:.4f}_{lat_max:.4f}.tif
    expected_filename = (
        f"etopo_{bounds[0]:.4f}_{bounds[1]:.4f}_{bounds[2]:.4f}_{bounds[3]:.4f}.tif"
    )
    expected_path = tmp_path / expected_filename

    # Since we mocked rasterio.open, the *write* to cache also uses the mock.
    # So the file won't actually exist on disk unless we mock the write differently or use a real file for cache write.
    # But we can verify the write call.
    # The service calls rasterio.open(path, 'w', ...)

    # Check if open was called with 'w' mode
    calls = mock_open.call_args_list
    # We expect at least:
    # 1. open(ETOPO_URL) (read)
    # 2. open(ETOPO_URL) (read again for transform? - yes in current impl)
    # 3. open(cache_path, 'w', ...) (write)

    write_call_found = False
    for call in calls:
        args, kwargs = call
        if len(args) > 0 and str(args[0]) == str(expected_path) and args[1] == "w":
            write_call_found = True
            break

    assert write_call_found, "Cache file write not triggered"
