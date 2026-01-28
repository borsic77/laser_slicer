import numpy as np
import pytest
import rasterio.transform
from shapely.geometry import Polygon

from core.utils.dem import mosaic_and_crop


def test_mosaic_and_crop_downsampling(monkeypatch):
    # Test that down_sample factor is passed to merge

    def fake_merge(src_files, res=None):
        return np.zeros((1, 5, 5)), rasterio.transform.from_origin(0, 10, 2, 2)

    from collections import namedtuple

    Bounds = namedtuple("Bounds", ["left", "bottom", "right", "top"])

    class DummySrc:
        res = (5, 5)
        bounds = Bounds(0, 0, 10, 10)
        transform = rasterio.transform.from_origin(0, 10, 5, 5)
        width = 2
        height = 2
        count = 1
        dtypes = ["float32"]
        crs = "EPSG:4326"
        nodata = -32768

        def close(self):
            pass

        def read(self, *args, **kwargs):
            return np.zeros((1, 2, 2))

    monkeypatch.setattr("rasterio.open", lambda *a, **k: DummySrc())
    monkeypatch.setattr("core.utils.dem.merge", fake_merge)
    monkeypatch.setattr(
        "core.utils.dem.from_bounds",
        lambda *a, **k: rasterio.windows.Window(0, 0, 5, 5),
    )

    merge_calls = []

    def spy_merge(src_files, res=None, **kwargs):
        merge_calls.append(res)
        return np.ones((1, 10, 10)), rasterio.transform.from_origin(0, 10, 1, 1)

    monkeypatch.setattr("core.utils.dem.merge", spy_merge)

    mosaic_and_crop(["dummy.tif"], (0, 0, 10, 10), downsample_factor=2)

    assert len(merge_calls) == 1
    assert merge_calls[0] == (10, 10)
