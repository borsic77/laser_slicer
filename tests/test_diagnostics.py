from pathlib import Path

import numpy as np
import pytest

from core.utils.diagnostics import save_elevation_visualization


def test_save_elevation_visualization(tmp_path):
    # Create dummy data
    data = np.linspace(-500, 2000, 100 * 100).reshape(100, 100)
    # Add a hole
    data[50:60, 50:60] = np.nan

    output_path = tmp_path / "test_diag.png"

    save_elevation_visualization(data, None, None, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
