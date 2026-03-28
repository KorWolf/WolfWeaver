import json
from pathlib import Path

from src.reconstruction_modes import get_reconstruction_mode_values
from src.score_modes import get_score_mode_values


def load_config(config_path: Path) -> dict:
    """
    Load the JSON config file and validate required fields.

    This is used by the CLI path and by any code that reads config
    directly from disk.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    # These keys are required by the current pipeline.
    required_keys = [
        "source_image",
        "palette_file",
        "frame_count",
        "save_debug_tables",
        "create_gif",
        "gif_frame_duration_ms",
        "gif_output_name",
        "frame_prefix",
        "reconstruction_mode",
        "score_mode",
        "random_seed",
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Config file is missing required keys: {missing_keys}")

    # Validate reconstruction mode against the central registry.
    valid_reconstruction_modes = get_reconstruction_mode_values()
    if config["reconstruction_mode"] not in valid_reconstruction_modes:
        raise ValueError(
            f"Invalid reconstruction_mode: {config['reconstruction_mode']}. "
            f"Valid options: {valid_reconstruction_modes}"
        )

    # Validate score mode against the central registry.
    valid_score_modes = get_score_mode_values()
    if config["score_mode"] not in valid_score_modes:
        raise ValueError(
            f"Invalid score_mode: {config['score_mode']}. "
            f"Valid options: {valid_score_modes}"
        )

    return config