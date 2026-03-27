import time
from pathlib import Path
from typing import List

from src.animate import create_gif_from_frames
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
from src.palette import (
    build_palette_dataframe,
    export_palette_to_csv,
    load_palette_from_json,
    rotate_palette_colors,
)
from src.reconstruct import reconstruct_image_from_assignments, save_image_array


def run_single_rotation(
    rotation_index: int,
    image_array,
    color_stats_df,
    base_palette_data: dict,
    total_pixels: int,
    save_debug_tables: bool = False,
) -> Path:
    """
    Run one full pipeline pass for a rotated palette order and save one frame.

    Returns:
        Path to the saved PNG frame
    """
    rotated_colors = rotate_palette_colors(base_palette_data["colors"], rotation_index)

    rotated_palette_data = {
        "name": f"{base_palette_data.get('name', 'palette')}_rot_{rotation_index:03d}",
        "colors": rotated_colors,
    }

    palette_df = build_palette_dataframe(
        palette_data=rotated_palette_data,
        total_pixels=total_pixels,
    )

    if int(palette_df["Quota"].sum()) != total_pixels:
        raise ValueError("Palette quota sanity check failed during rotation run.")

    difference_df = build_difference_matrix(
        source_df=color_stats_df,
        palette_df=palette_df,
    )

    long_difference_df = build_long_difference_table(
        source_df=color_stats_df,
        palette_df=palette_df,
    )

    assignment_df = build_assignment_table(
        source_df=color_stats_df,
        palette_df=palette_df,
        long_difference_df=long_difference_df,
    )

    assignment_summary_df = build_assignment_summary(assignment_df)

    output_image_array = reconstruct_image_from_assignments(
        image_array=image_array,
        assignment_df=assignment_df,
    )

    frame_path = Path(f"output/frames/frame_{rotation_index:03d}.png")
    save_image_array(output_image_array, frame_path)

    # Optional per-frame debug outputs
    if save_debug_tables:
        palette_csv = Path(f"output/debug/palette_rot_{rotation_index:03d}.csv")
        difference_preview_csv = Path(f"output/debug/difference_rot_{rotation_index:03d}.csv")
        long_difference_preview_csv = Path(f"output/debug/difference_long_rot_{rotation_index:03d}.csv")
        assignment_csv = Path(f"output/debug/assignment_rot_{rotation_index:03d}.csv")
        assignment_summary_csv = Path(f"output/debug/assignment_summary_rot_{rotation_index:03d}.csv")

        export_palette_to_csv(palette_df, palette_csv)
        export_difference_preview(difference_df, str(difference_preview_csv), num_rows=200)
        export_long_difference_preview(long_difference_df, str(long_difference_preview_csv), num_rows=1000)
        export_assignment_table(assignment_df, str(assignment_csv))
        export_assignment_summary(assignment_summary_df, str(assignment_summary_csv))

    print(f"Rotation {rotation_index:03d}: frame saved to {frame_path}")

    return frame_path


def main() -> None:
    start_time = time.perf_counter()

    print("Tapestry pipeline starting...")

    image_path = Path("input/source_images/birthOfVenus.png")
    palette_path = Path("input/palettes/example_palette.json")

    # Convention:
    # frame_count controls how many rotations to generate.
    # Set it equal to the palette size to generate the full cycle.
    frame_count = 10

    # Convention:
    # save_debug_tables=False keeps output lighter.
    # Set to True if you want CSV/debug files for every rotation.
    save_debug_tables = False

    # Convention:
    # GIF creation is optional but enabled by default here.
    create_gif = True
    gif_duration_ms = 150

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

    image_array = load_image(image_path)
    height, width = get_image_dimensions(image_array)
    total_pixels = width * height

    print(f"Image size: {width} x {height}")
    print(f"Total pixels: {total_pixels}")

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

    base_palette_data = load_palette_from_json(palette_path)

    if frame_count > len(base_palette_data["colors"]):
        raise ValueError(
            f"frame_count ({frame_count}) cannot exceed palette size ({len(base_palette_data['colors'])})"
        )

    print(f"Base palette name: {base_palette_data.get('name', 'unnamed_palette')}")
    print(f"Palette size: {len(base_palette_data['colors'])}")
    print(f"Generating {frame_count} rotated frame(s)...")

    frame_paths: List[Path] = []

    for rotation_index in range(frame_count):
        frame_start = time.perf_counter()

        frame_path = run_single_rotation(
            rotation_index=rotation_index,
            image_array=image_array,
            color_stats_df=color_stats_df,
            base_palette_data=base_palette_data,
            total_pixels=total_pixels,
            save_debug_tables=save_debug_tables,
        )

        frame_end = time.perf_counter()
        frame_duration = frame_end - frame_start

        print(f"Rotation {rotation_index:03d} completed in {frame_duration:.2f}s")

        frame_paths.append(frame_path)

    if create_gif:
        gif_path = Path("output/gifs/rotating_palette.gif")
        create_gif_from_frames(
            frame_paths=frame_paths,
            output_path=gif_path,
            duration_ms=gif_duration_ms,
        )
        print(f"GIF saved to {gif_path}")

    print(f"\nSaved source stats to: {source_stats_csv}")
    print(f"Generated {len(frame_paths)} frame(s).")

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    minutes = int(total_duration // 60)
    seconds = total_duration % 60

    print(f"\nTotal runtime: {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()