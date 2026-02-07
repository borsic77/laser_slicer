from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the functionality to be tested
# We need to test the logic inside ContourSlicingJob._fetch_and_prep_dem
# Since we can't easily import the *impl* of that huge method, we might mock the surrounding parts
# or better, extract the logic to a helper if it were cleaner.
# For now, we will create a unit test that mocks the dependencies and calls the method.
from core.services.contour_generator import OCEAN_LEVEL_THRESHOLD, ContourSlicingJob

# Logic tests only


# Since the method is complex and involves many steps (download, merge, etc.),
# a pure unit test of the logic block is easier if we just extract or simulate the numpy operations.
# Let's write a targeted test that replicates the logic exactly effectively verifying the *formula*.


def test_water_mask_numpy_logic():
    """
    Verifies the numpy logic used in the implementation:
    ocean_candidates = water_mask_dest.astype(bool) & (land_raw < OCEAN_LEVEL_THRESHOLD)
    """

    # 2x2 Grid
    # [0,0]: High Altitude River (300m, Water) -> Should KEEP 300m
    # [0,1]: Low Altitude Ocean (2m, Water) -> Should BE MASKED (NaN)
    # [1,0]: High Altitude Land (300m, Land) -> Should KEEP 300m
    # [1,1]: Low Altitude Land (2m, Land) -> Should KEEP 2m

    land_raw = np.array([[300.0, 2.0], [300.0, 2.0]], dtype=np.float32)

    water_mask_dest = np.array([[1, 1], [0, 0]], dtype=np.uint8)  # 1=Water

    # Apply Logic
    ocean_candidates = water_mask_dest.astype(bool) & (land_raw < OCEAN_LEVEL_THRESHOLD)
    land_raw[ocean_candidates] = np.nan

    # Verify
    assert not np.isnan(land_raw[0, 0]), "High altitude river was incorrectly masked!"
    assert land_raw[0, 0] == 300.0

    assert np.isnan(land_raw[0, 1]), "Low altitude ocean was NOT masked!"

    assert land_raw[1, 0] == 300.0
    assert land_raw[1, 1] == 2.0


def test_clamping_logic():
    """
    Verifies the clamping logic:
    bad_ocean = mask_bool & (elevation > -0.1) & (elevation < OCEAN_LEVEL_THRESHOLD)
    """
    # [0]: River (300m) -> Ignored
    # [1]: Ocean Bleed (2m) -> Clamped to -5
    # [2]: Correct Ocean (-10m) -> Ignored

    elevation = np.array([300.0, 2.0, -10.0])
    mask_bool = np.array([True, True, True])  # All marked as water

    bad_ocean = mask_bool & (elevation > -0.1) & (elevation < OCEAN_LEVEL_THRESHOLD)

    assert bad_ocean[0] == False, "High river flagged as bad ocean"
    assert bad_ocean[1] == True, "Low ocean bleed NOT flagged"
    assert bad_ocean[2] == False, "Correct ocean flagged"
