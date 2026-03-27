import json
from pathlib import Path
from typing import List

import pandas as pd


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a hex color like '#A1B2C3' into an (R, G, B) tuple.
    """
    normalized = hex_color.strip().lstrip("#")

    if len(normalized) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")

    r = int(normalized[0:2], 16)
    g = int(normalized[2:4], 16)
    b = int(normalized[4:6], 16)

    return (r, g, b)


def calculate_saturation(rgb: tuple[int, int, int]) -> int:
    """
    Saturation used in your Excel method:
    max(R, G, B) - min(R, G, B)
    """
    return max(rgb) - min(rgb)


def calculate_lightness(rgb: tuple[int, int, int]) -> float:
    """
    Lightness used in your Excel method:
    average of R, G, B
    """
    return sum(rgb) / 3.0


def load_palette_from_json(palette_path: Path) -> dict:
    """
    Load a palette JSON file.

    Expected structure:
    {
        "name": "palette_name",
        "colors": ["#RRGGBB", ...]
    }
    """
    if not palette_path.exists():
        raise FileNotFoundError(f"Palette file not found: {palette_path}")

    with palette_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "colors" not in data or not isinstance(data["colors"], list):
        raise ValueError("Palette JSON must contain a 'colors' list.")

    if len(data["colors"]) == 0:
        raise ValueError("Palette must contain at least one color.")

    return data


def calculate_even_quotas(total_pixels: int, palette_size: int) -> List[int]:
    """
    Divide total pixels as evenly as possible across the palette colors.
    """
    if palette_size <= 0:
        raise ValueError("palette_size must be greater than 0")

    base_quota = total_pixels // palette_size
    remainder = total_pixels % palette_size

    quotas = [base_quota] * palette_size

    for index in range(remainder):
        quotas[index] += 1

    return quotas


def build_palette_dataframe(palette_data: dict, total_pixels: int) -> pd.DataFrame:
    """
    Build a DataFrame containing the ordered replacement palette and quotas.

    Output columns:
    - Order
    - Hex
    - R
    - G
    - B
    - S
    - L
    - Quota
    """
    colors = palette_data["colors"]
    quotas = calculate_even_quotas(total_pixels=total_pixels, palette_size=len(colors))

    rows = []

    for index, hex_color in enumerate(colors):
        rgb = hex_to_rgb(hex_color)
        r, g, b = rgb
        s = calculate_saturation(rgb)
        l = calculate_lightness(rgb)

        rows.append(
            {
                "Order": index,
                "Hex": hex_color.upper(),
                "R": r,
                "G": g,
                "B": b,
                "S": s,
                "L": round(l, 4),
                "Quota": quotas[index],
            }
        )

    df = pd.DataFrame(rows)
    return df


def export_palette_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the palette DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)