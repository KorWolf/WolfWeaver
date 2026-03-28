from pathlib import Path

from src.color_stats import build_color_stats_dataframe, extract_color_frequencies
from src.image_io import load_image


def extract_top_image_palette_colors(
    image_path: Path,
    color_count: int,
) -> list[str]:
    """
    Extract the most frequent colors directly from the source image.

    Rules:
    - colors are ranked by descending pixel frequency
    - ties remain stable because build_color_stats_dataframe already
      sorts by Frequency descending, then Hex ascending
    """
    if color_count < 1:
        raise ValueError("Image-derived palette color count must be at least 1.")

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    image_array = load_image(image_path)
    color_frequencies = extract_color_frequencies(image_array)
    color_stats_df = build_color_stats_dataframe(color_frequencies)

    if color_count > len(color_stats_df):
        color_count = len(color_stats_df)

    return color_stats_df.head(color_count)["Hex"].tolist()