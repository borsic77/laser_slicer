import numpy as np
import pytest
import rasterio
from shapely.geometry import Polygon

from core.utils.contour_ops import generate_contours


def test_generate_contours_downsampling():
    """Test that resolution_scale reduces the data size."""
    # Create a 100x100 DEM
    data = np.random.rand(100, 100) * 100
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)

    # Run with downsampling
    # This won't easily expose the internal downsampling, but we can check if it runs without error
    # and maybe inspect logs if we could capture them.
    # Ideally we'd mock the internal meshgrid prep, but for now we just ensure it doesn't crash.
    contours = generate_contours(
        data, data, transform, interval=10, resolution_scale=0.5
    )
    assert isinstance(contours, list)


def test_generate_contours_early_filtering():
    """Test that min_area_deg2 filters out small polygons."""
    # Create a simple hill: peak at (50, 50), height 100
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = 100 * np.exp(-((X - 5) ** 2 + (Y - 5) ** 2))

    # Add some noise (tiny bumps)
    noise = np.random.rand(100, 100) * 5
    Z_noisy = Z + noise

    transform = rasterio.Affine(0.1, 0, 0, 0, -0.1, 10)

    # Generate with aggressive filtering
    # 1e-10 is the default. Let's try a larger filter.
    # In this coord system (0-10), area units are "units squared".
    # A single pixel is 0.1 * 0.1 = 0.01 area.
    # Let's filter anything < 0.05 (approx 5 pixels)
    contours_filtered = generate_contours(
        Z_noisy, Z_noisy, transform, interval=10, min_area_deg2=0.05
    )

    # Generate without filtering (pass 0)
    contours_raw = generate_contours(
        Z_noisy, Z_noisy, transform, interval=10, min_area_deg2=0.0
    )

    # Filtered should have fewer or equal features
    # (Actually much fewer because of the noise)
    print(f"Raw: {len(contours_raw)}, Filtered: {len(contours_filtered)}")
    # We can't strictly assert < because noise might create large blobs,
    # but practically for random noise it works.
    # Let's check that we don't have extremely tiny polygons in filtered

    from shapely.geometry import shape

    for c in contours_filtered:
        geom = shape(c["geometry"])
        if not geom.is_empty:
            # It's a MultiPolygon or Polygon. Check sub-geometries.
            if geom.geom_type == "MultiPolygon":
                for g in geom.geoms:
                    assert g.area >= 0.05
            elif geom.geom_type == "Polygon":
                assert geom.area >= 0.05


def test_mosaic_downsampling(tmp_path):
    """Test that mosaic_and_crop respects downsample_factor."""
    import rasterio
    from rasterio.transform import from_origin

    from core.utils.dem import mosaic_and_crop

    # Create valid dummy GeoTIFFs
    d1 = tmp_path / "tile1.tif"

    # 100x100 raster
    data = np.random.rand(100, 100).astype(np.float32)
    # Transform: 1m pixel size, TL at (0, 100)
    transform = from_origin(0, 100, 1, 1)

    with rasterio.open(
        d1,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    bounds = (0, 0, 100, 100)  # Full coverage

    # Native resolution
    arr_full, _ = mosaic_and_crop([str(d1)], bounds, downsample_factor=1)

    # Downsample 10x
    arr_small, _ = mosaic_and_crop([str(d1)], bounds, downsample_factor=10)

    print(f"Full shape: {arr_full.shape}, Small shape: {arr_small.shape}")

    # Allow for some rounding differences in cropping
    assert arr_small.shape[0] < arr_full.shape[0]
    assert arr_small.shape[1] < arr_full.shape[1]


def test_sample_elevation(tmp_path):
    """Test sample_elevation with windowed read."""
    import rasterio
    from rasterio.transform import from_origin

    from core.utils.dem import sample_elevation

    d1 = tmp_path / "sample.tif"

    # 10x10 raster
    data = np.zeros((10, 10), dtype=np.float32)
    # Set a specific pixel
    data[5, 5] = 123.45

    # Transform: 1 deg pixel size (huge), TL at (0, 10)
    transform = from_origin(0, 10, 1, 1)

    # Write as uncompressed for simplicity in test, but code handles vsigzip
    # We can just path the path directly if not using vsigzip prefix in test,
    # BUT the function hardcodes /vsigzip/.
    # We need to mock rasterio.open or create a gzip file.
    # Actually, simpler to just write a .tif and NOT gzip it,
    # but the function prepends /vsigzip/.
    # So we MUST gzip it or mock the function.
    # Let's mock the path in the function call? No, it's hardcoded.
    # Let's gzip the file.

    import gzip

    # Write TIFF first
    with rasterio.open(
        d1,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    # Gzip it
    d1_gz = tmp_path / "sample.tif.gz"
    with open(d1, "rb") as f_in:
        with gzip.open(d1_gz, "wb") as f_out:
            import shutil

            shutil.copyfileobj(f_in, f_out)

    # Function takes absolute path to .tif.gz
    # The function inside dem.py does `f"/vsigzip/{dem_path}"`.
    # So we pass the absolute path to the gz file.

    # Coordinate for pixel (5, 5).
    # Origin (0, 10). Resolution (1, -1).
    # x = 0 + (5 + 0.5) * 1 = 5.5
    # y = 10 + (5 + 0.5) * -1 = 4.5

    val = sample_elevation(4.5, 5.5, str(d1_gz))
    assert abs(val - 123.45) < 0.001
