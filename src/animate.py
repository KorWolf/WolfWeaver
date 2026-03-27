from pathlib import Path
from typing import List

import imageio.v2 as imageio


def create_gif_from_frames(
    frame_paths: List[Path],
    output_path: Path,
    duration_ms: int = 150,
) -> None:
    """
    Create a GIF from a list of frame image paths.

    Args:
        frame_paths: ordered list of PNG frame paths
        output_path: target GIF path
        duration_ms: time per frame in milliseconds
    """
    if len(frame_paths) == 0:
        raise ValueError("No frame paths provided for GIF creation.")

    images = [imageio.imread(frame_path) for frame_path in frame_paths]
    duration_seconds = duration_ms / 1000.0

    imageio.mimsave(output_path, images, duration=duration_seconds, loop=0)