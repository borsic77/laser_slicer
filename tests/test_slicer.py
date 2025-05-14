import os

import numpy as np
import pytest

from core.utils.slicer import download_srtm_tiles_for_bounds, mosaic_and_crop


@pytest.fixture
def switzerland_bounds():
    # Covers Yverdon-les-Bains and surroundings
    return (6.5, 46.7, 6.7, 46.9)


def test_download_srtm_tiles_for_bounds(switzerland_bounds):
    paths = download_srtm_tiles_for_bounds(switzerland_bounds)
    assert isinstance(paths, list)
    assert all(os.path.exists(p) for p in paths)
    assert any("srtm" in os.path.basename(p) for p in paths)


def test_mosaic_and_crop(switzerland_bounds):
    paths = download_srtm_tiles_for_bounds(switzerland_bounds)
    array, transform = mosaic_and_crop(paths, switzerland_bounds)
    assert isinstance(array, np.ndarray)
    assert array.ndim == 2
    assert array.size > 0
    assert transform is not None
