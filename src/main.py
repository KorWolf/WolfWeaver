from pathlib import Path

from src.assign import (
    build_assignment_summary,
    build_assignment_table,
    export_assignment_summary,
    export_assignment_table,
)
from src.color_stats import (
    build_color_stats_dataframe,
    export_color_stats_to_csv,
    extract_color_frequencies,
)
from src.difference import (
    build_difference_matrix,
    build_long_difference_table,
    export_difference_preview,
    export_long_difference_preview,
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

    # Difference matrix
    print("\nBuilding difference matrix...")
    difference_df = build_difference_matrix(
        source_df=color_stats_df,
        palette_df=palette_df,
    )

    difference_preview_csv = Path("output/tables/difference_matrix_preview.csv")
    export_difference_preview(
        difference_df=difference_df,
        output_path=str(difference_preview_csv),
        num_rows=200,
    )

    print("Wide difference matrix preview saved.")

    print("Building long-form difference table...")
    long_difference_df = build_long_difference_table(
        source_df=color_stats_df,
        palette_df=palette_df,
    )

    long_difference_preview_csv = Path("output/tables/difference_long_preview.csv")
    export_long_difference_preview(
        long_df=long_difference_df,
        output_path=str(long_difference_preview_csv),
        num_rows=1000,
    )

    print("Long-form difference preview saved.")

    print(f"Wide difference shape: {difference_df.shape}")
    print(f"Long difference shape: {long_difference_df.shape}")

    # Assignment step
    print("\nBuilding assignment table...")
    assignment_df = build_assignment_table(
        source_df=color_stats_df,
        palette_df=palette_df,
        long_difference_df=long_difference_df,
    )

    assignment_csv = Path("output/tables/assignment_table.csv")
    export_assignment_table(assignment_df, str(assignment_csv))

    assignment_summary_df = build_assignment_summary(assignment_df)
    assignment_summary_csv = Path("output/tables/assignment_summary.csv")
    export_assignment_summary(assignment_summary_df, str(assignment_summary_csv))

    print("Assignment table saved.")
    print("Assignment summary saved.")

    print(f"\nAssignment rows: {len(assignment_df)}")
    print("\nAssignment summary preview:")
    print(assignment_summary_df.to_string(index=False))

    print(f"\nSaved source stats to: {source_stats_csv}")
    print(f"Saved replacement palette to: {palette_csv}")
    print(f"Saved difference preview to: {difference_preview_csv}")
    print(f"Saved long difference preview to: {long_difference_preview_csv}")
    print(f"Saved assignment table to: {assignment_csv}")
    print(f"Saved assignment summary to: {assignment_summary_csv}")


if __name__ == "__main__":
    main()