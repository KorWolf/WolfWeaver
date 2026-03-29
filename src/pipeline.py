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
from src.palette_source import (
    extract_clustered_main_palette_colors,
    extract_top_frequency_low_variety_palette_colors,
    extract_top_image_palette_colors,
)
from src.postprocess import apply_postprocess
from src.reconstruct import reconstruct_image_from_assignments, save_image_array
from src.run_manager import (
    create_run_directory,
    create_run_subdirectories,
    save_config_snapshot,
)


# ============================================================
# Progress update helper
# ============================================================

def build_progress_update(
    *,
    status: str,
    stage_label: str,
    run_dir=None,
    config_snapshot_path=None,
    source_stats_csv=None,
    frame_paths=None,
    gif_path=None,
    image_width=None,
    image_height=None,
    total_pixels=None,
    unique_source_colors=None,
    frame_count: int = 0,
    frame_timings_seconds=None,
    total_runtime_seconds: float = 0.0,
    palette_name=None,
    palette_size=None,
    reconstruction_mode=None,
    score_mode=None,
    postprocess_mode=None,
    completed_frames: int = 0,
) -> dict[str, Any]:
    """
    Build a consistent progress update payload for the UI.
    """
    return {
        "status": status,
        "stage_label": stage_label,
        "run_dir": run_dir,
        "config_snapshot_path": config_snapshot_path,
        "source_stats_csv": source_stats_csv,
        "frame_paths": list(frame_paths or []),
        "gif_path": gif_path,
        "image_width": image_width,
        "image_height": image_height,
        "total_pixels": total_pixels,
        "unique_source_colors": unique_source_colors,
        "frame_count": frame_count,
        "frame_timings_seconds": list(frame_timings_seconds or []),
        "total_runtime_seconds": total_runtime_seconds,
        "palette_name": palette_name,
        "palette_size": palette_size,
        "reconstruction_mode": reconstruction_mode,
        "score_mode": score_mode,
        "postprocess_mode": postprocess_mode,
        "completed_frames": completed_frames,
    }


# ============================================================
# Palette resolution helpers
# ============================================================
# A run can get its palette from either:
# - palette_colors already present in runtime config
# - image-derived palette settings
# - palette_file on disk
# ============================================================

def build_image_derived_palette_data(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build palette data from image-derived palette settings.

    This work is intentionally done inside the pipeline so Flask can
    redirect to the run page before heavier palette extraction begins.
    """
    image_path = Path(config["source_image"])
    image_palette_method = str(config.get("image_palette_method"))
    image_palette_count = int(config.get("image_palette_count"))
    random_seed = int(config["random_seed"])

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    if image_palette_method == "clustered_main_nearest":
        palette_colors = extract_clustered_main_palette_colors(
            image_path=image_path,
            color_count=image_palette_count,
            preserve_darkest=True,
            preserve_lightest=True,
            min_color_distance=28.0,
            random_seed=random_seed,
            representative_mode="nearest_real",
            selection_mode="standard",
        )
    elif image_palette_method == "clustered_main_frequency":
        palette_colors = extract_clustered_main_palette_colors(
            image_path=image_path,
            color_count=image_palette_count,
            preserve_darkest=True,
            preserve_lightest=True,
            min_color_distance=28.0,
            random_seed=random_seed,
            representative_mode="most_frequent_real",
            selection_mode="standard",
        )
    elif image_palette_method == "clustered_main_balanced":
        palette_colors = extract_clustered_main_palette_colors(
            image_path=image_path,
            color_count=image_palette_count,
            preserve_darkest=True,
            preserve_lightest=True,
            min_color_distance=28.0,
            random_seed=random_seed,
            representative_mode="most_frequent_real",
            selection_mode="balanced",
        )
    elif image_palette_method == "clustered_main_diverse":
        palette_colors = extract_clustered_main_palette_colors(
            image_path=image_path,
            color_count=image_palette_count,
            preserve_darkest=True,
            preserve_lightest=True,
            min_color_distance=28.0,
            random_seed=random_seed,
            representative_mode="most_frequent_real",
            selection_mode="diverse",
        )
    elif image_palette_method == "clustered_main_low_variety":
        palette_colors = extract_clustered_main_palette_colors(
            image_path=image_path,
            color_count=image_palette_count,
            preserve_darkest=True,
            preserve_lightest=True,
            min_color_distance=32.0,
            random_seed=random_seed,
            representative_mode="most_frequent_real",
            selection_mode="low_variety",
        )
    elif image_palette_method == "top_frequency_low_variety":
        palette_colors = extract_top_frequency_low_variety_palette_colors(
            image_path=image_path,
            color_count=image_palette_count,
            preserve_darkest=True,
            preserve_lightest=True,
            min_color_distance=26.0,
        )
    elif image_palette_method == "top_frequency":
        palette_colors = extract_top_image_palette_colors(
            image_path=image_path,
            color_count=image_palette_count,
        )
    else:
        raise ValueError(f"Unsupported image-derived palette method: {image_palette_method}")

    return {
        "name": f"image_{image_palette_method}_{len(palette_colors)}",
        "colors": palette_colors,
    }


def resolve_palette_data(config: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve the palette data used for the run.

    Priority:
    1. Use palette_colors directly if present in config
    2. Build image-derived palette if requested
    3. Otherwise load palette JSON from palette_file
    """
    palette_colors = config.get("palette_colors")
    palette_source = str(config.get("palette_source", "manual"))

    if palette_colors is not None:
        if not isinstance(palette_colors, list) or len(palette_colors) == 0:
            raise ValueError("palette_colors must be a non-empty list of hex colors.")

        return {
            "name": config.get("palette_name", "ui_palette"),
            "colors": palette_colors,
        }

    if palette_source == "image":
        return build_image_derived_palette_data(config)

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
    palette_source = str(config.get("palette_source", "manual"))

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

    if palette_source not in {"manual", "image"}:
        raise ValueError("palette_source must be either 'manual' or 'image'.")

    if palette_source == "image":
        image_palette_method = str(config.get("image_palette_method"))
        image_palette_count = config.get("image_palette_count")

        valid_image_methods = {
            "clustered_main_nearest",
            "clustered_main_frequency",
            "clustered_main_balanced",
            "clustered_main_diverse",
            "clustered_main_low_variety",
            "top_frequency",
            "top_frequency_low_variety",
        }

        if image_palette_method not in valid_image_methods:
            raise ValueError("Invalid image_palette_method for image-derived palette.")

        if image_palette_count is None or int(image_palette_count) < 1:
            raise ValueError("image_palette_count must be at least 1 for image-derived palette.")


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
    - running
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
    palette_source = str(config.get("palette_source", "manual"))

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    # Report the palette step first so the user can tell what setup phase is active.
    if palette_source == "image":
        yield build_progress_update(
            status="running",
            stage_label="Building image-derived palette",
            frame_count=frame_count,
            total_runtime_seconds=time.perf_counter() - start_time,
            reconstruction_mode=reconstruction_mode,
            score_mode=score_mode,
            postprocess_mode=postprocess_mode,
        )
    else:
        yield build_progress_update(
            status="running",
            stage_label="Using manual palette",
            frame_count=frame_count,
            total_runtime_seconds=time.perf_counter() - start_time,
            reconstruction_mode=reconstruction_mode,
            score_mode=score_mode,
            postprocess_mode=postprocess_mode,
        )

    # Resolve palette data inside the worker so heavier image-derived
    # palette methods do not block the Flask request path.
    base_palette_data = resolve_palette_data(config)

    # Update config before snapshot so the saved settings show the actual
    # palette colors and final palette name used for the run.
    config["palette_colors"] = list(base_palette_data["colors"])
    config["palette_name"] = str(base_palette_data.get("name", "unnamed_palette"))

    palette_size = len(base_palette_data["colors"])

    yield build_progress_update(
        status="running",
        stage_label="Preparing run output folders",
        frame_count=frame_count,
        total_runtime_seconds=time.perf_counter() - start_time,
        palette_name=base_palette_data.get("name", "unnamed_palette"),
        palette_size=palette_size,
        reconstruction_mode=reconstruction_mode,
        score_mode=score_mode,
        postprocess_mode=postprocess_mode,
    )

    # Prepare run output folders and save a config snapshot for traceability.
    run_dir = create_run_directory()
    run_subdirs = create_run_subdirectories(run_dir)
    config_snapshot_path = save_config_snapshot(config, run_dir)

    yield build_progress_update(
        status="running",
        stage_label="Loading source image",
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        frame_count=frame_count,
        total_runtime_seconds=time.perf_counter() - start_time,
        palette_name=base_palette_data.get("name", "unnamed_palette"),
        palette_size=palette_size,
        reconstruction_mode=reconstruction_mode,
        score_mode=score_mode,
        postprocess_mode=postprocess_mode,
    )

    # Load source image and derive basic image stats.
    image_array = load_image(image_path)
    height, width = get_image_dimensions(image_array)
    total_pixels = width * height

    yield build_progress_update(
        status="running",
        stage_label="Counting source colors",
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        image_width=width,
        image_height=height,
        total_pixels=total_pixels,
        frame_count=frame_count,
        total_runtime_seconds=time.perf_counter() - start_time,
        palette_name=base_palette_data.get("name", "unnamed_palette"),
        palette_size=palette_size,
        reconstruction_mode=reconstruction_mode,
        score_mode=score_mode,
        postprocess_mode=postprocess_mode,
    )

    color_freq = extract_color_frequencies(image_array)
    sum_counts = sum(color_freq.values())

    if total_pixels != sum_counts:
        raise ValueError("Sanity check failed: total pixel count does not match summed frequencies.")

    yield build_progress_update(
        status="running",
        stage_label="Preparing source color table",
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        image_width=width,
        image_height=height,
        total_pixels=total_pixels,
        unique_source_colors=len(color_freq),
        frame_count=frame_count,
        total_runtime_seconds=time.perf_counter() - start_time,
        palette_name=base_palette_data.get("name", "unnamed_palette"),
        palette_size=palette_size,
        reconstruction_mode=reconstruction_mode,
        score_mode=score_mode,
        postprocess_mode=postprocess_mode,
    )

    # Build and export source color stats once for the whole run.
    color_stats_df = build_color_stats_dataframe(color_freq)
    source_stats_csv = run_subdirs["tables"] / "source_color_stats.csv"
    export_color_stats_to_csv(color_stats_df, str(source_stats_csv))

    frame_paths: list[Path] = []
    frame_timings: list[float] = []

    # Generate each requested frame and report the active frame number before work starts.
    for rotation_index in range(frame_count):
        current_frame_number = rotation_index + 1

        yield build_progress_update(
            status="running",
            stage_label=f"Generating frame {current_frame_number} of {frame_count}",
            run_dir=run_dir,
            config_snapshot_path=config_snapshot_path,
            source_stats_csv=source_stats_csv,
            frame_paths=frame_paths,
            image_width=width,
            image_height=height,
            total_pixels=total_pixels,
            unique_source_colors=len(color_freq),
            frame_count=frame_count,
            frame_timings_seconds=frame_timings,
            total_runtime_seconds=time.perf_counter() - start_time,
            palette_name=base_palette_data.get("name", "unnamed_palette"),
            palette_size=palette_size,
            reconstruction_mode=reconstruction_mode,
            score_mode=score_mode,
            postprocess_mode=postprocess_mode,
            completed_frames=len(frame_paths),
        )

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

        yield build_progress_update(
            status="running",
            stage_label=f"Finished frame {current_frame_number} of {frame_count}",
            run_dir=run_dir,
            config_snapshot_path=config_snapshot_path,
            source_stats_csv=source_stats_csv,
            frame_paths=frame_paths,
            image_width=width,
            image_height=height,
            total_pixels=total_pixels,
            unique_source_colors=len(color_freq),
            frame_count=frame_count,
            frame_timings_seconds=frame_timings,
            total_runtime_seconds=time.perf_counter() - start_time,
            palette_name=base_palette_data.get("name", "unnamed_palette"),
            palette_size=palette_size,
            reconstruction_mode=reconstruction_mode,
            score_mode=score_mode,
            postprocess_mode=postprocess_mode,
            completed_frames=len(frame_paths),
        )

    gif_path: Path | None = None

    # Optional GIF creation after all frames are complete.
    if create_gif:
        yield build_progress_update(
            status="running",
            stage_label="Generating GIF",
            run_dir=run_dir,
            config_snapshot_path=config_snapshot_path,
            source_stats_csv=source_stats_csv,
            frame_paths=frame_paths,
            image_width=width,
            image_height=height,
            total_pixels=total_pixels,
            unique_source_colors=len(color_freq),
            frame_count=frame_count,
            frame_timings_seconds=frame_timings,
            total_runtime_seconds=time.perf_counter() - start_time,
            palette_name=base_palette_data.get("name", "unnamed_palette"),
            palette_size=palette_size,
            reconstruction_mode=reconstruction_mode,
            score_mode=score_mode,
            postprocess_mode=postprocess_mode,
            completed_frames=len(frame_paths),
        )

        gif_path = run_dir / "gifs" / gif_output_name
        create_gif_from_frames(
            frame_paths=frame_paths,
            output_path=gif_path,
            duration_ms=gif_duration_ms,
        )

    end_time = time.perf_counter()
    total_duration = end_time - start_time

    yield build_progress_update(
        status="completed",
        stage_label="Completed",
        run_dir=run_dir,
        config_snapshot_path=config_snapshot_path,
        source_stats_csv=source_stats_csv,
        frame_paths=frame_paths,
        gif_path=gif_path,
        image_width=width,
        image_height=height,
        total_pixels=total_pixels,
        unique_source_colors=len(color_freq),
        frame_count=frame_count,
        frame_timings_seconds=frame_timings,
        total_runtime_seconds=total_duration,
        palette_name=base_palette_data.get("name", "unnamed_palette"),
        palette_size=palette_size,
        reconstruction_mode=reconstruction_mode,
        score_mode=score_mode,
        postprocess_mode=postprocess_mode,
        completed_frames=len(frame_paths),
    )


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