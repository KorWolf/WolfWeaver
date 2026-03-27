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
from src.config_loader import load_config
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
from src.run_manager import (
    create_run_directory,
    create_run_subdirectories,
    save_config_snapshot,
)


def run_single_rotation(
    rotation_index: int,
    image_array,
    color_stats_df,
    base_palette_data: dict,
    total_pixels: int,
    frame_prefix: str,
    reconstruction_mode: str,
    random_seed: int,
    run_subdirs: dict[str, Path],
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
        reconstruction_mode=reconstruction_mode,
        random_seed=random_seed + rotation_index,
    )

    frame_path = run_subdirs["frames"] / f"{frame_prefix}_{rotation_index:03d}.png"
    save_image_array(output_image_array, frame_path)

    if save_debug_tables:
        palette_csv = run_subdirs["debug"] / f"palette_rot_{rotation_index:03d}.csv"
        difference_preview_csv = run_subdirs["debug"] / f"difference_rot_{rotation_index:03d}.csv"
        long_difference_preview_csv = run_subdirs["debug"] / f"difference_long_rot_{rotation_index:03d}.csv"
        assignment_csv = run_subdirs["debug"] / f"assignment_rot_{rotation_index:03d}.csv"
        assignment_summary_csv = run_subdirs["debug"] / f"assignment_summary_rot_{rotation_index:03d}.csv"

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

    config = load_config(Path("config.json"))

    image_path = Path(config["source_image"])
    palette_path = Path(config["palette_file"])
    frame_count = int(config["frame_count"])
    save_debug_tables = bool(config["save_debug_tables"])
    create_gif = bool(config["create_gif"])
    gif_duration_ms = int(config["gif_frame_duration_ms"])
    gif_output_name = str(config["gif_output_name"])
    frame_prefix = str(config["frame_prefix"])
    reconstruction_mode = str(config["reconstruction_mode"])
    random_seed = int(config["random_seed"])

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    if not palette_path.exists():
        raise FileNotFoundError(f"Palette file not found: {palette_path}")

    run_dir = create_run_directory()
    run_subdirs = create_run_subdirectories(run_dir)
    config_snapshot_path = save_config_snapshot(config, run_dir)

    print(f"Run directory: {run_dir}")

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
    source_stats_csv = run_subdirs["tables"] / "source_color_stats.csv"
    export_color_stats_to_csv(color_stats_df, str(source_stats_csv))

    base_palette_data = load_palette_from_json(palette_path)

    if frame_count > len(base_palette_data["colors"]):
        raise ValueError(
            f"frame_count ({frame_count}) cannot exceed palette size ({len(base_palette_data['colors'])})"
        )

    print(f"Base palette name: {base_palette_data.get('name', 'unnamed_palette')}")
    print(f"Palette size: {len(base_palette_data['colors'])}")
    print(f"Reconstruction mode: {reconstruction_mode}")
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
            frame_prefix=frame_prefix,
            reconstruction_mode=reconstruction_mode,
            random_seed=random_seed,
            run_subdirs=run_subdirs,
            save_debug_tables=save_debug_tables,
        )

        frame_end = time.perf_counter()
        frame_duration = frame_end - frame_start

        print(f"Rotation {rotation_index:03d} completed in {frame_duration:.2f}s")
        frame_paths.append(frame_path)

    if create_gif:
        gif_path = run_subdirs["gifs"] / gif_output_name
        create_gif_from_frames(
            frame_paths=frame_paths,
            output_path=gif_path,
            duration_ms=gif_duration_ms,
        )
        print(f"GIF saved to {gif_path}")

    print(f"\nSaved source stats to: {source_stats_csv}")
    print(f"Saved config snapshot to: {config_snapshot_path}")
    print(f"Generated {len(frame_paths)} frame(s).")

    end_time = time.perf_counter()
    total_duration = end_time - start_time
    minutes = int(total_duration // 60)
    seconds = total_duration % 60

    print(f"\nTotal runtime: {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()