from pathlib import Path

from src.color_stats import (
    build_color_stats_dataframe,
    export_color_stats_to_csv,
    extract_color_frequencies,
)
from src.image_io import get_image_dimensions, load_image
from src.palette import build_palette_dataframe, export_palette_to_csv, load_palette_from_json


def main() -> None:
    print("Tapestry pipeline starting...")

    image_path = Path("input/source_images/birthOfVenus.png")
    palette_path = Path("input/palettes/example_palette.json")

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    if not palette_path.exists():
        raise FileNotFoundError(f"Palette file not found: {palette_path}")

    required_dirs = [
        Path("output/frames"),
        Path("output/gifs"),
        Path("output/tables"),
        Path("output/debug"),
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # Load image
    image_array = load_image(image_path)

    # Dimensions
    height, width = get_image_dimensions(image_array)
    total_pixels = width * height

    print(f"Image size: {width} x {height}")
    print(f"Total pixels: {total_pixels}")

    # Source color stats
    color_freq = extract_color_frequencies(image_array)
    print(f"Unique source colors: {len(color_freq)}")

    sum_counts = sum(color_freq.values())
    print(f"Sum of source frequencies: {sum_counts}")

    if total_pixels != sum_counts:
        raise ValueError("Sanity check failed: total pixel count does not match summed frequencies.")

    print("Source color sanity check passed.")

    color_stats_df = build_color_stats_dataframe(color_freq)
    source_stats_csv = Path("output/tables/source_color_stats.csv")
    export_color_stats_to_csv(color_stats_df, str(source_stats_csv))

    # Replacement palette stats
    palette_data = load_palette_from_json(palette_path)
    palette_df = build_palette_dataframe(palette_data=palette_data, total_pixels=total_pixels)

    palette_csv = Path("output/tables/replacement_palette.csv")
    export_palette_to_csv(palette_df, palette_csv)

    print(f"\nPalette name: {palette_data.get('name', 'unnamed_palette')}")
    print(f"Replacement colors: {len(palette_df)}")
    print(f"Sum of replacement quotas: {int(palette_df['Quota'].sum())}")

    if int(palette_df["Quota"].sum()) != total_pixels:
        raise ValueError("Palette quota sanity check failed: quotas do not sum to total pixels.")

    print("Palette quota sanity check passed.")

    print("\nReplacement palette preview:")
    print(palette_df.head(10).to_string(index=False))

    print(f"\nSaved source stats to: {source_stats_csv}")
    print(f"Saved replacement palette to: {palette_csv}")


if __name__ == "__main__":
    main()