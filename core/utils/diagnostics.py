import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_elevation_visualization(
    elevation_data: np.ndarray,
    masked_data: np.ndarray | None,
    bathy_data: np.ndarray | None,
    filename: str | Path,
) -> None:
    """
    Save a visualization of the elevation data to a PNG file.

    Args:
        elevation_data: The full elevation array (SRTM/Merged).
        masked_data: Elevation array with NaNs/Masked values (optional).
        bathy_data: Bathymetry array (optional).
        filename: Output path.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot main elevation
    # Use a terrestrial colormap
    if elevation_data is not None:
        im = ax.imshow(elevation_data, cmap="terrain", origin="upper", alpha=0.8)
        plt.colorbar(im, ax=ax, label="Elevation (m)")

    # Overlay masked data if provided (e.g. to show holes)
    # We can plot it with a different map or just rely on main elevation if it's already masked.
    # If masked_data is a boolean mask, we could overlay it.
    # If it's the array with nans, imshow handles nans as transparent usually.

    # Plot 0m contour (coastline)
    if elevation_data is not None:
        try:
            ax.contour(elevation_data, levels=[0], colors="red", linewidths=1.0)
        except Exception:
            pass  # No zero crossing

    ax.set_title("Elevation Diagnostic")
    ax.axis("off")

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
