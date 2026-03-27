from pathlib import Path

from src.image_io import load_image, get_image_dimensions
from src.color_stats import (
    build_color_stats_dataframe,
    export_color_stats_to_csv,
    extract_color_frequencies,
)


def main() -> None:
    print("Tapestry pipeline starting...")

    # Convention:
    # Change this filename if you use a different test image.
    image_path = Path("input/source_images/birthOfVenus.png")

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    # Ensure output folders exist
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

    # Get dimensions
    height, width = get_image_dimensions(image_array)
    print(f"Image size: {width} x {height}")

    # Count colors
    color_freq = extract_color_frequencies(image_array)
    print(f"Unique colors: {len(color_freq)}")

    # Sanity check
    total_pixels = width * height
    sum_counts = sum(color_freq.values())

    print(f"Total pixels: {total_pixels}")
    print(f"Sum of frequencies: {sum_counts}")

    if total_pixels != sum_counts:
        raise ValueError("Sanity check failed: total pixel count does not match summed frequencies.")

    print("Sanity check passed.")

    # Build table
    color_stats_df = build_color_stats_dataframe(color_freq)

    print("\nTop 10 rows:")
    print(color_stats_df.head(10).to_string(index=False))

    # Export CSV
    output_csv_path = Path("output/tables/source_color_stats.csv")
    export_color_stats_to_csv(color_stats_df, str(output_csv_path))

    print(f"\nSaved color stats table to: {output_csv_path}")


if __name__ == "__main__":
    main()

    