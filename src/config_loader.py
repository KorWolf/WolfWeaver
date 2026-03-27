import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    """
    Load the JSON config file and validate required fields.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

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
        "random_seed"
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Config file is missing required keys: {missing_keys}")

    valid_modes = ["scanline", "random_seeded"]
    if config["reconstruction_mode"] not in valid_modes:
        raise ValueError(
            f"Invalid reconstruction_mode: {config['reconstruction_mode']}. "
            f"Valid options: {valid_modes}"
        )

    return config