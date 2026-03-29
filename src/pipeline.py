import time
from pathlib import Path
from typing import Any, Generator

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
from src.postprocess import apply_postprocess
from src.reconstruct import reconstruct_image_from_assignments, save_image_array
from src.run_manager import (
    create_run_directory,
    create_run_subdirectories,
    save_config_snapshot,
)


# ============================================================
# Palette resolution helpers
# ============================================================
# A run can get its palette from either:
# - palette_colors already present in runtime config
# - palette_file on disk
# ============================================================

def resolve_palette_data(config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve the palette data used for the run.

    Priority:
    1. Use palette_colors directly if present in config
    2. Otherwise load palette JSON from palette_file
    """
    palette_colors = config.get("palette_colors")

    if palette_colors is not None:
        if not isinstance(palette_colors, list) or len(palette_colors) == 0:
            raise ValueError("palette_colors must be a non-empty list of hex colors.")

        return {
            "name": config.get("palette_name", "ui_palette"),
            "colors": palette_colors,
        }

    palette_path = Path(config["palette_file"])
    if not palette_path.exists():
        raise FileNotFoundError(f"Palette file not found: {palette_path}")

    return load_palette_from_json(palette_path)


# ============================================================
# Runtime config validation
# ============================================================

def validate_runtime_config(config: dict[str, Any]) -> None:
    """
    Validate runtime-only numeric settings before processing starts.
    """
    frame_count = int(config["frame_count"])
    gif_frame_duration_ms = int(config["gif_frame_duration_ms"])
    random_seed = int(config["random_seed"])
    postprocess_mode = str(config.get("postprocess_mode", "none"))

    if frame_count < 1:
        raise ValueError("frame_count must be at least 1.")

    if gif_frame_duration_ms < 1:
        raise ValueError("gif_frame_duration_ms must be at least 1.")

    if random_seed < 0:
        raise ValueError("random_seed must be 0 or greater.")

    if postprocess_mode not in {"none", "coherence_basic", "coherence_edge_aware"}:
        raise ValueError(
            "postprocess_mode must be one of: "
            "none, coherence_basic, coherence_edge_aware."
        )


# ============================================================
# Single rotation runner
# ============================================================
# One "rotation" means:
# - rotate the palette ordering
# - build quotas for that rotated palette
# - calculate source/replacement differences
# - build the assignment table
# - reconstruct the image using the selected reconstruction mode
# - optionally apply post-processing
# - save one frame
# ============================================================

def run_single_rotation(
    rotation_index: int,
    image_array,
    color_stats_df,
    base_palette_data: dict,
    total_pixels: int,
    frame_prefix: str,
    reconstruction_mode: str,
    score_mode: str,
    postprocess_mode: str,
    random_seed: int,
    run_subdirs: dict[str, Path],
    save_debug_tables: bool = False,
) -> Path:
    """
    Run one full palette rotation and save one output frame.

    Returns:
    - Path to the saved frame PNG
    """
    palette_size = len(base_palette_data["colors"])
    effective_rotation_index = rotation_index % palette_size

    rotated_colors = rotate_palette_colors(
        base_palette_data["colors"],
        effective_rotation_index,
    )

    rotated_palette_data = {
        "name": f"{base_palette_data.get('name', 'palette')}_rot_{rotation_index:03d}",
        "colors": rotated_colors,
    }

    # Build the ordered replacement palette with quotas for this rotation.
    palette_df = build_palette_dataframe(
        palette_data=rotated_palette_data,
        total_pixels=total_pixels,
    )

    # Sanity check: total quota must equal total image pixels.
    if int(palette_df["Quota"].sum()) != total_pixels:
        raise ValueError("Palette quota sanity check failed during rotation run.")

    # Build score tables using the selected score mode.
    difference_df = build_difference_matrix(
        source_df=color_stats_df,
        palette_df=palette_df,
        score_mode=score_mode,
    )

    long_difference_df = build_long_difference_table(
        source_df=color_stats_df,
        palette_df=palette_df,
        score_mode=score_mode,
    )

    # Build the assignment table that maps source colors to replacement colors.
    assignment_df = build_assignment_table(
        source_df=color_stats_df,
        palette_df=palette_df,
        long_difference_df=long_difference_df,
    )

    assignment_summary_df = build_assignment_summary(assignment_df)

    # Reconstruct the image using the selected reconstruction mode.
    output_image_array = reconstruct_image_from_assignments(
        image_array=image_array,
        assignment_df=assignment_df,
        reconstruction_mode=reconstruction_mode,
        random_seed=random_seed + rotation_index,
    )

    # Apply optional post-processing after reconstruction.
    output_image_array = apply_postprocess(
        source_image_array=image_array,
        reconstructed_image_array=output_image_array,
        palette_colors=rotated_palette_data["colors"],
        postprocess_mode=postprocess_mode,
    )

    frame_path = run_subdirs["frames"] / f"{frame_prefix}_{rotation_index:03d}.png"
    save_image_array(output_image_array, frame_path)

    # Optional debug exports for inspection.
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

    return frame_path


# ============================================================
# Streaming pipeline
# ============================================================
# This generator is used by the Flask job system.
# It yields progress updates as work completes.
# ============================================================

def run_pipeline_stream(config: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    """
    Run the full pipeline and yield progress updates.

    Yield stages:
    - started
    - running (after each completed frame)
    - completed
    """
    start_time = time.perf_counter()

    validate_runtime_config(config)

    image_path = Path(config["source_image"])
    frame_count = int(config["frame_count"])
    save_debug_tables = bool(config["save_debug_tables"])
    create_gif = bool(config["create_gif"])
    gif_duration_ms = int(config["gif_frame_duration_ms"])
    gif_output_name = str(config["gif_output_name"])
    frame_prefix = str(config["frame_prefix"])
    reconstruction_mode = str(config["reconstruction_mode"])
    score_mode = str(config.get("score_mode", "basic_rgb_sl"))
    postprocess_mode = str(config.get("postprocess_mode", "none"))
    random_seed = int(config["random_seed"])

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    # Prepare run output folders and save a config snapshot for traceability.
    run_dir = create_run_directory()
    run_subdirs = create_run_subdirectories(run_dir)
    config_snapshot_path = save_config_snapshot(config, run_dir)

    base_palette_data = resolve_palette_data(config)
    palette_size = len(base_palette_data["colors"])

    # Load source image and derive basic image stats.
    image_array = load_image(image_path)
    height, width = get_image_dimensions(image_array)
    total_pixels = width * height

    color_freq = extract_color_frequencies(image_array)
    sum_counts = sum(color_freq.values())

    if total_pixels != sum_counts:
        raise ValueError("Sanity check failed: total pixel count does not match summed frequencies.")

    # Build and export source color stats once for the whole run.
    color_stats_df = build_color_stats_dataframe(color_freq)
    source_stats_csv = run_subdirs["tables"] / "source_color_stats.csv"
    export_color_stats_to_csv(color_stats_df, str(source_stats_csv))

    frame_paths: list[Path] = []
    frame_timings: list[float] = []

    # Initial progress update so the UI can render metadata before frame 1 finishes.
    yield {
        "status": "started",
        "run_dir": run_dir,
        "config_snapshot_path": config_snapshot_path,
        "source_stats_csv": source_stats_csv,
        "frame_paths": [],
        "gif_path": None,
        "image_width": width,
        "image_height": height,
        "total_pixels": total_pixels,
        "unique_source_colors": len(color_freq),
        "frame_count": frame_count,
        "frame_timings_seconds": [],
        "total_runtime_seconds": 0.0,
        "palette_name": base_palette_data.get("name", "unnamed_palette"),
        "palette_size": palette_size,
        "reconstruction_mode": reconstruction_mode,
        "score_mode": score_mode,
        "postprocess_mode": postprocess_mode,
        "completed_frames": 0,
    }

    # Build one frame for each requested rotation.
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
            score_mode=score_mode,
            postprocess_mode=postprocess_mode,
            random_seed=random_seed,
            run_subdirs=run_subdirs,
            save_debug_tables=save_debug_tables,
        )

        frame_end = time.perf_counter()
        frame_duration = frame_end - frame_start

        frame_paths.append(frame_path)
        frame_timings.append(frame_duration)

        yield {
            "status": "running",
            "run_dir": run_dir,
            "config_snapshot_path": config_snapshot_path,
            "source_stats_csv": source_stats_csv,
            "frame_paths": list(frame_paths),
            "gif_path": None,
            "image_width": width,
            "image_height": height,
            "total_pixels": total_pixels,
            "unique_source_colors": len(color_freq),
            "frame_count": frame_count,
            "frame_timings_seconds": list(frame_timings),
            "total_runtime_seconds": time.perf_counter() - start_time,
            "palette_name": base_palette_data.get("name", "unnamed_palette"),
            "palette_size": palette_size,
            "reconstruction_mode": reconstruction_mode,
            "score_mode": score_mode,
            "postprocess_mode": postprocess_mode,
            "completed_frames": len(frame_paths),
        }

    gif_path: Path | None = None

    # Optional GIF creation after all frames are complete.
    if create_gif:
        gif_path = run_dir / "gifs" / gif_output_name
        create_gif_from_frames(
            frame_paths=frame_paths,
            output_path=gif_path,
            duration_ms=gif_duration_ms,
        )

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    yield {
        "status": "completed",
        "run_dir": run_dir,
        "config_snapshot_path": config_snapshot_path,
        "source_stats_csv": source_stats_csv,
        "frame_paths": list(frame_paths),
        "gif_path": gif_path,
        "image_width": width,
        "image_height": height,
        "total_pixels": total_pixels,
        "unique_source_colors": len(color_freq),
        "frame_count": frame_count,
        "frame_timings_seconds": list(frame_timings),
        "total_runtime_seconds": total_duration,
        "palette_name": base_palette_data.get("name", "unnamed_palette"),
        "palette_size": palette_size,
        "reconstruction_mode": reconstruction_mode,
        "score_mode": score_mode,
        "postprocess_mode": postprocess_mode,
        "completed_frames": len(frame_paths),
    }


# ============================================================
# Non-streaming wrapper
# ============================================================
# This is used by the CLI entry point.
# It runs the same pipeline but only returns the final result.
# ============================================================

def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run the pipeline to completion and return the final result dict.
    """
    final_result = None

    for update in run_pipeline_stream(config):
        final_result = update

    if final_result is None:
        raise RuntimeError("Pipeline did not produce any result.")

    return final_result