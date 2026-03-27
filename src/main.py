from pathlib import Path

from src.config_loader import load_config
from src.pipeline import run_pipeline


def main() -> None:
    print("Tapestry pipeline starting...")

    config = load_config(Path("config.json"))
    result = run_pipeline(config)

    print(f"Run directory: {result['run_dir']}")
    print(f"Image size: {result['image_width']} x {result['image_height']}")
    print(f"Total pixels: {result['total_pixels']}")
    print(f"Unique source colors: {result['unique_source_colors']}")
    print(f"Palette name: {result['palette_name']}")
    print(f"Palette size: {result['palette_size']}")
    print(f"Reconstruction mode: {result['reconstruction_mode']}")
    print(f"Generated {result['frame_count']} frame(s).")
    print(f"Saved source stats to: {result['source_stats_csv']}")
    print(f"Saved config snapshot to: {result['config_snapshot_path']}")

    if result["gif_path"] is not None:
        print(f"GIF saved to: {result['gif_path']}")

    for index, duration in enumerate(result["frame_timings_seconds"]):
        print(f"Rotation {index:03d} completed in {duration:.2f}s")

    total_duration = float(result["total_runtime_seconds"])
    minutes = int(total_duration // 60)
    seconds = total_duration % 60

    print(f"\nTotal runtime: {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()