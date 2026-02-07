import logging
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from PIL import Image

from core.services.contour_generator import ContourSlicingJob
from core.utils.diagnostics import save_elevation_visualization
from core.utils.geocoding import compute_bounds_from_center_and_size

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Diagnose shoreline slicing issues by generating intermediate rasters."

    def add_arguments(self, parser):
        parser.add_argument(
            "--lat", type=float, default=43.21415, help="Center Latitude"
        )
        parser.add_argument(
            "--lon", type=float, default=5.32680, help="Center Longitude"
        )
        parser.add_argument(
            "--width", type=float, default=1327.0, help="Width in meters"
        )
        parser.add_argument(
            "--height", type=float, default=1259.0, help="Height in meters"
        )
        parser.add_argument(
            "--output", type=str, default="data/diagnostics", help="Output directory"
        )

    def handle(self, *args, **options):
        lat = options["lat"]
        lon = options["lon"]
        width = options["width"]
        height = options["height"]
        output_dir = Path(options["output"])

        output_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(f"Diagnosing area at {lat}, {lon} ({width}m x {height}m)")

        # Calculate bounds
        bounds = compute_bounds_from_center_and_size(lat, lon, width, height)
        self.stdout.write(f"Bounds: {bounds}")

        # Initialize Job (mocking parameters not strictly needed for DEM fetch)
        job = ContourSlicingJob(
            bounds=bounds,
            height_per_layer=10,
            num_layers=10,
            simplify=0,
            substrate_size_mm=100,
            layer_thickness_mm=1,
            center=(lon, lat),
            smoothing=0,
            min_area=0,
            min_feature_width_mm=0,
            include_bathymetry=True,  # Important for testing shoreline
        )

        # 1. Fetch OSM Map Tiles (Reference)
        self.stdout.write("Fetching OSM Map Reference...")
        try:
            from core.utils.osm_tiles import (
                TILE_PROVIDERS,
                fetch_osm_static_image,
                generate_water_mask_from_tiles,
            )

            # Use Voyager NoLabels for the mask
            provider_name = "cartodb_voyager_nolabels"
            water_color = TILE_PROVIDERS[provider_name]["water_color"]

            osm_img, transform, crs = fetch_osm_static_image(
                bounds, zoom=15, provider=provider_name
            )
            osm_img.save(output_dir / "diag_1_osm_reference.png")

            # 1.5 Generate Water Mask
            self.stdout.write("Generating Water Mask from Tiles...")
            # Use specific water color for this provider
            water_mask = generate_water_mask_from_tiles(
                osm_img, water_color=water_color
            )
            # Save mask (convert bool to uint8 0-255)
            mask_img = Image.fromarray((water_mask * 255).astype(np.uint8))
            mask_img.save(output_dir / "diag_1b_water_mask.png")

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Failed to fetch OSM tiles or mask: {e}")
            )

        # 2. Fetch Land Only (SRTM/ALTI3D)
        self.stdout.write("Fetching Land DEM (SRTM)...")
        from core.utils.dem import download_elevation_tiles_for_bounds, mosaic_and_crop

        srtm_paths = download_elevation_tiles_for_bounds(bounds)
        land_raw, _ = mosaic_and_crop(srtm_paths, bounds)
        save_elevation_visualization(
            land_raw, None, None, output_dir / "diag_2_land_raw.png"
        )

        # 3. Fetch Bathymetry Only
        self.stdout.write("Fetching Bathymetry...")
        from core.services.bathymetry_service import BathymetryFetcher

        bf = BathymetryFetcher()
        try:
            bathy_raw = bf.fetch_elevation_for_bounds(bounds)
            save_elevation_visualization(
                bathy_raw, None, None, output_dir / "diag_3_bathy_raw.png"
            )
        except Exception as e:
            logger.error(f"Failed to fetch bathymetry: {e}")
            bathy_raw = None

        # 4. Fetch Merged (via Job)
        self.stdout.write("Fetching Merged DEM...")
        elevation, masked_elevation, transform = job._fetch_and_prep_dem()
        save_elevation_visualization(
            elevation, masked_elevation, None, output_dir / "diag_4_merged_final.png"
        )

        self.stdout.write(
            self.style.SUCCESS(f"Diagnosis complete. Files saved to {output_dir}")
        )
