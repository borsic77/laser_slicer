import pytest

from core.utils.download_clip_elevation_tiles import (
    download_srtm_tiles_for_bounds,
    ensure_tile_downloaded,
    get_srtm_tile_path,
    is_antimeridian_crossing,
)


@pytest.fixture
def switzerland_bounds():
    # Covers Yverdon-les-Bains and surroundings
    return (6.5, 46.7, 6.7, 46.9)


@pytest.fixture
def antimeridian_bounds():
    # Crosses the antimeridian (example: from 179E to -179W)
    return (179.5, -16.0, -179.5, -15.5)


def test_download_srtm_tiles_for_bounds(monkeypatch, switzerland_bounds):
    # Patch _download_tiles to avoid real downloads
    called = {}

    def fake_download_tiles(bounds):
        called["bounds"] = bounds
        return [f"/fake/path/srtm_{b}.hgt.gz" for b in bounds]

    monkeypatch.setattr(
        "core.utils.download_clip_elevation_tiles._download_tiles", fake_download_tiles
    )
    paths = download_srtm_tiles_for_bounds(switzerland_bounds)
    assert isinstance(paths, list)
    assert all(p.startswith("/fake/path/srtm_") for p in paths)


def test_download_srtm_tiles_for_antimeridian(monkeypatch, antimeridian_bounds):
    # Track calls to check recursion and splitting
    calls = []

    def fake_download_tiles(bounds):
        calls.append(bounds)
        return [f"/fake/path/srtm_{bounds[0]}.hgt.gz"]

    monkeypatch.setattr(
        "core.utils.download_clip_elevation_tiles._download_tiles", fake_download_tiles
    )

    # Only return True for the initial call, then False after splitting
    original_bounds = antimeridian_bounds

    def fake_is_antimeridian_crossing(bounds):
        return bounds == original_bounds

    monkeypatch.setattr(
        "core.utils.download_clip_elevation_tiles.is_antimeridian_crossing",
        fake_is_antimeridian_crossing,
    )

    paths = download_srtm_tiles_for_bounds(antimeridian_bounds)
    # Should get two calls (splitting east and west of 180°)
    assert len(paths) == 2
    assert all(p.startswith("/fake/path/srtm_") for p in paths)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ((0.0, 0.0, 10.0, 10.0), False),
        ((170.0, -10.0, -170.0, 10.0), True),
        ((-180.0, -10.0, 179.0, 10.0), False),
    ],
)
def test_is_antimeridian_crossing(bounds, expected):
    assert is_antimeridian_crossing(bounds) is expected


def test_get_srtm_tile_path(tmp_path):
    path1 = get_srtm_tile_path(46.8, 6.6, tmp_path)
    assert path1 == tmp_path / "N46E006.hgt.gz"

    path2 = get_srtm_tile_path(-16.2, -179.9, tmp_path)
    assert path2 == tmp_path / "S17W180.hgt.gz"


@pytest.mark.parametrize(
    "lat,lon,expected_bounds,filename",
    [
        (46.8, 6.6, (6, 46, 7, 47), "N46E006.hgt.gz"),
        (-16.2, -179.9, (-180, -17, -179, -16), "S17W180.hgt.gz"),
    ],
)
def test_ensure_tile_downloaded(monkeypatch, tmp_path, lat, lon, expected_bounds, filename):
    def fake_download(bounds):
        assert bounds == expected_bounds
        return [str(tmp_path / filename)]

    monkeypatch.setattr(
        "core.utils.download_clip_elevation_tiles.download_srtm_tiles_for_bounds",
        fake_download,
    )

    result = ensure_tile_downloaded(lat, lon)
    assert result == tmp_path / filename
