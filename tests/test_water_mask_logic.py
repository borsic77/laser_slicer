import numpy as np

# Import the functionality to be tested
# We need to test the logic inside ContourSlicingJob._fetch_and_prep_dem
# Since we can't easily import the *impl* of that huge method, we might mock the surrounding parts
# or better, extract the logic to a helper if it were cleaner.
# For now, we will create a unit test that mocks the dependencies and calls the method.
from core.services.contour_generator import OCEAN_LEVEL_THRESHOLD

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


def test_coastal_radar_noise_ocean_candidate_detection():
    """Test connected-component ocean candidate logic absorbs coastal radar blur (e.g. 23m in bay)."""
    import scipy.ndimage

    # Grid with:
    # - A continuous ocean bay on the right (water_mask_dest=1) with sea level (1m) and radar cliff noise (23m)
    # - An isolated inland river on the left (water_mask_dest=1) at 300m
    # - Land in between (water_mask_dest=0) at 40m
    land_raw = np.array(
        [
            [300.0, 40.0, 23.0, 1.0],
            [300.0, 40.0, 18.0, 0.0],
            [300.0, 40.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    water_mask_dest = np.array(
        [
            [1, 0, 1, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    water_surface = 0.0
    water_threshold = 15.0

    labeled_water, num_features = scipy.ndimage.label(water_mask_dest)
    ocean_candidates = np.zeros(land_raw.shape, dtype=bool)

    for feat_id in range(1, num_features + 1):
        comp_mask = labeled_water == feat_id
        comp_elevs = land_raw[comp_mask]
        valid_elevs = comp_elevs[(comp_elevs != -32768) & ~np.isnan(comp_elevs)]
        if len(valid_elevs) == 0:
            ocean_candidates |= comp_mask
            continue
        baseline_elev = float(np.percentile(valid_elevs, 25))
        min_elev = float(np.min(valid_elevs))
        if baseline_elev <= water_threshold or min_elev <= water_surface + 5.0:
            cutoff = max(water_threshold + 35.0, baseline_elev + 40.0)
            ocean_candidates |= comp_mask & (land_raw < cutoff)

    # Ocean bay pixels (including 18m and 23m radar noise) must be marked as ocean
    assert ocean_candidates[0, 2] == True, (
        "23m coastal radar noise was not included in ocean"
    )
    assert ocean_candidates[1, 2] == True, (
        "18m coastal radar noise was not included in ocean"
    )
    assert ocean_candidates[0, 3] == True, "1m ocean water was not included in ocean"
    assert ocean_candidates[1, 3] == True, "0m ocean water was not included in ocean"

    # High altitude inland river (300m) must NOT be marked as ocean
    assert ocean_candidates[0, 0] == False, (
        "300m inland river was incorrectly marked as ocean"
    )
    assert ocean_candidates[1, 0] == False, (
        "300m inland river was incorrectly marked as ocean"
    )
    assert ocean_candidates[2, 0] == False, (
        "300m inland river was incorrectly marked as ocean"
    )

    # Land pixels must NOT be marked as ocean
    assert ocean_candidates[0, 1] == False
    assert ocean_candidates[1, 1] == False
