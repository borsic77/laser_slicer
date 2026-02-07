import time
from unittest.mock import MagicMock

import pytest

from core.services.contour_generator import ContourSlicingJob


def test_parallel_osm_fetching_speed(monkeypatch):
    # Mock the granular fetch functions to sleep 1 second each
    def slow_fetch(*args, **kwargs):
        time.sleep(1.0)
        return MagicMock(is_empty=True)

    # We need to mock the methods on the INSTANCE or the module where they are defined.
    # In ContourSlicingJob._fetch_osm_features, it calls:
    # self._prepare_osm_road_geoms
    # self._prepare_osm_feature_geom (twice, for buildings and waterways)

    # Let's monkeypatch ContourSlicingJob methods directly

    original_prep_road = ContourSlicingJob._prepare_osm_road_geoms
    original_prep_feat = ContourSlicingJob._prepare_osm_feature_geom

    try:

        def mock_prep_road(self, *args, **kwargs):
            time.sleep(1.0)
            return {}

        def mock_prep_feat(self, *args, **kwargs):
            time.sleep(1.0)
            return MagicMock(
                is_empty=True
            )  # Return something that isn't None but is empty geom

        ContourSlicingJob._prepare_osm_road_geoms = mock_prep_road
        ContourSlicingJob._prepare_osm_feature_geom = mock_prep_feat

        job = ContourSlicingJob(
            bounds=(0, 0, 1, 1),
            height_per_layer=10,
            num_layers=10,
            simplify=0,
            substrate_size_mm=100,
            layer_thickness_mm=3,
            center=(0.5, 0.5),
            smoothing=0,
            min_area=0,
            min_feature_width_mm=0,
            include_roads=True,
            include_buildings=True,
            include_waterways=True,
        )

        start_time = time.time()
        # calling private method for testing purpose
        job._fetch_osm_features(
            utm_bounds=(0, 0, 100, 100), proj_params=None, cx=0.5, cy=0.5
        )
        end_time = time.time()
        duration = end_time - start_time

        # If sequential: 1s (roads) + 1s (waterways) + 1s (buildings) = 3s
        # If parallel: ~1s total
        print(f"Duration: {duration:.2f}s")
        assert duration < 2.5, (
            "Fetching should be parallel (taking < 2.5s for 3x 1s tasks)"
        )

    finally:
        ContourSlicingJob._prepare_osm_road_geoms = original_prep_road
        ContourSlicingJob._prepare_osm_feature_geom = original_prep_feat
