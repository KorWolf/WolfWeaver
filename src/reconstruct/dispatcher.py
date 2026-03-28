import numpy as np
import pandas as pd

from .random_modes import (
    reconstruct_image_random_seeded,
    reconstruct_image_random_unseeded,
    reconstruct_image_weighted_random,
)
from .scanline import reconstruct_image_scanline
from .structured_modes import (
    reconstruct_image_block_seeded,
    reconstruct_image_checker_seeded,
)


def reconstruct_image_from_assignments(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
    reconstruction_mode: str = "scanline",
    random_seed: int = 42,
) -> np.ndarray:
    """
    Dispatch reconstruction based on the selected mode.
    """
    if reconstruction_mode == "scanline":
        return reconstruct_image_scanline(
            image_array=image_array,
            assignment_df=assignment_df,
        )

    if reconstruction_mode == "random_seeded":
        return reconstruct_image_random_seeded(
            image_array=image_array,
            assignment_df=assignment_df,
            random_seed=random_seed,
        )

    if reconstruction_mode == "checker_seeded":
        return reconstruct_image_checker_seeded(
            image_array=image_array,
            assignment_df=assignment_df,
            random_seed=random_seed,
        )

    if reconstruction_mode == "block_seeded":
        return reconstruct_image_block_seeded(
            image_array=image_array,
            assignment_df=assignment_df,
            random_seed=random_seed,
            block_size=4,
        )

    if reconstruction_mode == "random_unseeded":
        return reconstruct_image_random_unseeded(
            image_array=image_array,
            assignment_df=assignment_df,
        )

    if reconstruction_mode == "weighted_random":
        return reconstruct_image_weighted_random(
            image_array=image_array,
            assignment_df=assignment_df,
            random_seed=random_seed,
        )

    raise ValueError(f"Unsupported reconstruction_mode: {reconstruction_mode}")