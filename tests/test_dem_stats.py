from unittest.mock import patch

import numpy as np
import pytest

from core.utils.dem import get_elevation_stats_fast


def test_elevation_stats_fast():
    # Mock download_srtm_tiles_for_bounds to return a dummy file
    with patch(
        "core.utils.dem.download_srtm_tiles_for_bounds", return_value=["dummy.hgt.gz"]
    ) as mock_download:
        # Mock mosaic_and_crop to return a known array
        # Min = 100, Max = 1000
        mock_data = np.array([[100, 200], [500, 1000]], dtype=np.int16)
        with patch(
            "core.utils.dem.mosaic_and_crop", return_value=(mock_data, None)
        ) as mock_mosaic:
            min_val, max_val = get_elevation_stats_fast((0, 0, 1, 1))

            assert min_val == 100.0
            assert max_val == 1000.0

            # Verify calls
            mock_download.assert_called_with((0, 0, 1, 1))
            mock_mosaic.assert_called_with(
                ["dummy.hgt.gz"], (0, 0, 1, 1), downsample_factor=10
            )


def test_elevation_stats_fast_no_tiles():
    with patch(
        "core.utils.dem.download_srtm_tiles_for_bounds", return_value=[]
    ) as mock_download:
        min_val, max_val = get_elevation_stats_fast((0, 0, 1, 1))
        assert min_val == 0.0
        assert max_val == 0.0
