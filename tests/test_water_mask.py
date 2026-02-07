import numpy as np
import pytest
from PIL import Image

from core.utils.osm_tiles import generate_water_mask_from_tiles


def test_water_mask_generation():
    # Create a dummy image (20x20 to have space)
    # Background: Green (Land)
    # Patch: Blue (Water)

    width, height = 20, 20
    image = Image.new("RGB", (width, height), color=(0, 255, 0))  # Green

    pixels = image.load()
    water_color = (170, 211, 223)  # OSM Blue

    # Draw a 2x2 water patch at (1,1)
    pixels[1, 1] = water_color
    pixels[1, 2] = water_color
    pixels[2, 1] = water_color
    pixels[2, 2] = water_color

    # Generate mask
    mask = generate_water_mask_from_tiles(image, water_color=water_color)

    assert mask.shape == (height, width)
    assert mask.dtype == bool

    # Check water pixels
    assert mask[1, 1] == True
    assert mask[1, 2] == True

    # Check land pixels
    assert mask[0, 0] == False
    assert mask[5, 5] == False

    # Test tolerance (slightly different blue)
    pixels[5, 5] = (172, 213, 225)  # +2 difference
    mask_tol = generate_water_mask_from_tiles(image, water_color=water_color)
    assert mask_tol[5, 5] == True

    # Test Morphological Closing (Text/Artifact removal)
    # Create a block of water with a "hole" (land color) in the middle
    # 5x5 block of water
    # Extend image size for this test
    width, height = 20, 20
    image = Image.new("RGB", (width, height), color=(0, 255, 0))  # Green
    pixels = image.load()

    for x in range(10, 15):
        for y in range(10, 15):
            pixels[x, y] = water_color

    # Add a hole in the middle (simulating text)
    pixels[12, 12] = (0, 255, 0)  # Green Land

    mask_closed = generate_water_mask_from_tiles(image, water_color=water_color)

    # The hole should be closed (True = Water)
    # The hole should be closed (True = Water)
    assert mask_closed[12, 12] == True


from unittest.mock import MagicMock, patch

from core.services.contour_generator import ContourSlicingJob


def test_water_mask_clamping_logic():
    """
    Test that positive elevation values in water-masked areas are clamped to -5.0m.
    """
    bounds = (0, 0, 1, 1)
    # Instantiate job with minimal params
    job = ContourSlicingJob(
        bounds=bounds,
        height_per_layer=10,
        num_layers=10,
        simplify=0,
        substrate_size_mm=100,
        layer_thickness_mm=1,
        center=(0.5, 0.5),
        smoothing=0,
        min_area=0,
        min_feature_width_mm=0,
        include_bathymetry=True,
        use_osm_water_mask=True,
    )

    # Define our test data
    shape = (3, 3)

    # Land Raw (SRTM) - Real land at [0,0], Water at [1,1]
    land_raw = np.full(shape, np.nan)
    land_raw[0, 0] = 50.0

    # Bathy Raw - Deep ocean everywhere
    bathy_raw = np.full(shape, -100.0)

    # Water Mask - [1,1] is WATER. [0,0] is LAND.
    water_mask_src = np.zeros(shape, dtype=bool)
    water_mask_src[1, 1] = True  # Water

    # Merged result coming from merge_srtm_and_etopo
    # Let's simulate a "Vanishing Island" scenario:
    # At [1,1] (Water), the merged data has POSITIVE elevation (e.g. 5.0m) due to ETOPO error
    merged_elevation = np.full(shape, 50.0)  # Default land
    merged_elevation[1, 1] = 5.0  # BAD OCEAN!
    merged_elevation[2, 2] = -100.0  # Good Ocean

    # Start Patching
    # Context manager for multiple patches
    # Note: We must patch the SOURCE modules because contour_generator imports them locally!
    # Start Patching
    # Context manager for multiple patches
    # Note: We must patch the SOURCE modules because contour_generator imports them locally!
    with (
        patch("core.utils.dem.download_elevation_tiles_for_bounds"),
        patch(
            "core.utils.dem.mosaic_and_crop", return_value=(land_raw, MagicMock())
        ) as mock_mosaic,
        patch("core.services.bathymetry_service.BathymetryFetcher") as MockFetcher,
        patch(
            "core.utils.osm_tiles.fetch_osm_static_image",
            return_value=(MagicMock(), MagicMock(), "EPSG:3857"),
        ),
        patch(
            "core.utils.osm_tiles.generate_water_mask_from_tiles",
            return_value=water_mask_src,
        ),
        patch(
            "rasterio.warp.reproject",
            side_effect=lambda source, destination, **kwargs: np.copyto(
                destination, source
            ),
        ),
        patch(
            "core.utils.dem.merge_srtm_and_etopo", return_value=merged_elevation
        ) as mock_merge,
        patch("core.utils.dem.clean_srtm_dem", side_effect=lambda x, **kwargs: x),
        patch("core.utils.dem.fill_nans_in_dem", side_effect=lambda x: x),
    ):
        # Configure Bathymetry Fetcher Mock
        mock_fetcher_instance = MockFetcher.return_value
        mock_fetcher_instance.fetch_elevation_for_bounds.return_value = bathy_raw

        # Run the method
        elevation, _, _ = job._fetch_and_prep_dem()

        # 3. Assertions

        # [0,0] is Land, should match input (50.0)
        assert elevation[0, 0] == 50.0, "Land pixel should remain 50.0"

        # [2,2] is Good Ocean (-100.0), should match input
        assert elevation[2, 2] == -100.0, "Good ocean pixel should remain -100.0"

        # [1,1] is Bad Ocean (5.0), should be clamped to -5.0
        assert elevation[1, 1] == -5.0, (
            f"Bad ocean pixel was {elevation[1, 1]}, expected -5.0"
        )

        # Verify merge_srtm_and_etopo was called with the correct land_mask
        # land_mask should be the inverse of water_mask_src (since we mocked reproject to simple copy)
        # water_mask_src has True at [1,1]
        # land_mask should have False at [1,1] and True elsewhere
        args, kwargs = mock_merge.call_args
        assert "land_mask" in kwargs
        passed_mask = kwargs["land_mask"]
        assert passed_mask[1, 1] == False, (
            "Land mask should be False where Water mask is True"
        )
        assert passed_mask[0, 0] == True, (
            "Land mask should be True where Water mask is False"
        )
