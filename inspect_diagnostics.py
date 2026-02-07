import numpy as np
from PIL import Image


def analyze_image(path):
    print(f"--- Analyzing {path} ---")
    try:
        img = Image.open(path)
        arr = np.array(img)
        print(f"Shape: {arr.shape}, Dtype: {arr.dtype}")
        unique_vals = np.unique(arr)
        if len(unique_vals) > 20:
            print(f"Unique values (first 20): {unique_vals[:20]}")
            print(f"Range: {arr.min()} - {arr.max()}")
        else:
            print(f"Unique values: {unique_vals}")

        if len(arr.shape) == 3:  # RGB
            # Check for most common colors
            colors, counts = np.unique(
                arr.reshape(-1, arr.shape[2]), axis=0, return_counts=True
            )
            sorted_indices = np.argsort(-counts)
            print("Top 5 Colors:")
            for i in sorted_indices[:5]:
                print(f"  {colors[i]}: {counts[i]} pixels")

    except Exception as e:
        print(f"Error: {e}")


base = "data/diagnostics"
analyze_image(f"{base}/diag_1b_water_mask.png")
analyze_image(f"{base}/diag_1_osm_reference.png")
