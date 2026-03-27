import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


def create_run_directory(base_output_dir: Path = Path("output/runs")) -> Path:
    """
    Create a unique per-run output directory.

    Example:
    output/runs/run_20260326_184512/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"run_{timestamp}"

    suffix = 1
    original_run_dir = run_dir
    while run_dir.exists():
        run_dir = Path(f"{original_run_dir}_{suffix:02d}")
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def create_run_subdirectories(run_dir: Path) -> dict[str, Path]:
    """
    Create standard subdirectories inside a run directory.
    """
    subdirs = {
        "frames": run_dir / "frames",
        "gifs": run_dir / "gifs",
        "tables": run_dir / "tables",
        "debug": run_dir / "debug",
    }

    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return subdirs


def save_config_snapshot(config: dict, run_dir: Path) -> Path:
    """
    Save a copy of the config used for this run.
    """
    snapshot_path = run_dir / "config_snapshot.json"

    config_copy = deepcopy(config)
    config_copy["run_directory"] = str(run_dir)

    with snapshot_path.open("w", encoding="utf-8") as file:
        json.dump(config_copy, file, indent=4)

    return snapshot_path