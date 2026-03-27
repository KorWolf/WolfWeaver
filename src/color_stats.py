from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Convention:
# RGB color tuples are always ordered as (R, G, B)
RGBTuple = Tuple[int, int, int]


def extract_color_frequencies(image_array: np.ndarray) -> Dict[RGBTuple, int]:
    """
    Count how many times each RGB color appears in the image.

    Args:
        image_array: NumPy array with shape (height, width, 3)

    Returns:
        Dictionary mapping (R, G, B) tuples to pixel counts
    """
    pixels = image_array.reshape(-1, 3)
    pixel_tuples = [tuple(int(channel) for channel in pixel) for pixel in pixels]
    return dict(Counter(pixel_tuples))


def rgb_to_hex(rgb: RGBTuple) -> str:
    """
    Convert an (R, G, B) tuple into a hex string like '#A1B2C3'.
    """
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def calculate_saturation(rgb: RGBTuple) -> int:
    """
    Saturation used in your Excel method:
    max(R, G, B) - min(R, G, B)
    """
    return max(rgb) - min(rgb)


def calculate_lightness(rgb: RGBTuple) -> float:
    """
    Lightness used in your Excel method:
    average of R, G, B
    """
    return sum(rgb) / 3.0


def build_color_stats_dataframe(color_frequencies: Dict[RGBTuple, int]) -> pd.DataFrame:
    """
    Build a DataFrame containing per-color stats.

    Output columns:
    - Hex
    - Frequency
    - R
    - G
    - B
    - S
    - L
    """
    rows = []

    for rgb, frequency in color_frequencies.items():
        r, g, b = rgb
        s = calculate_saturation(rgb)
        l = calculate_lightness(rgb)

        rows.append(
            {
                "Hex": rgb_to_hex(rgb),
                "Frequency": frequency,
                "R": r,
                "G": g,
                "B": b,
                "S": s,
                "L": round(l, 4),
            }
        )

    df = pd.DataFrame(rows)

    # Convention:
    # Sort by Frequency descending first, then Hex ascending for stable output.
    df = df.sort_values(by=["Frequency", "Hex"], ascending=[False, True]).reset_index(drop=True)

    return df


def export_color_stats_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the color stats table to CSV.
    """
    df.to_csv(output_path, index=False)