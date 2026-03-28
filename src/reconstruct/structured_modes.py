from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.palette import hex_to_rgb

from .common import build_source_replacement_plan, rgb_to_hex


def build_checker_position_order(
    positions: List[Tuple[int, int]],
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Return positions in checkerboard-class order.

    Positions are split into two groups:
    - group 0: (row + col) % 2 == 0
    - group 1: (row + col) % 2 == 1

    Each group is shuffled independently using the provided RNG,
    then concatenated together.
    """
    checker_even: List[Tuple[int, int]] = []
    checker_odd: List[Tuple[int, int]] = []

    for row_index, column_index in positions:
        if (row_index + column_index) % 2 == 0:
            checker_even.append((row_index, column_index))
        else:
            checker_odd.append((row_index, column_index))

    rng.shuffle(checker_even)
    rng.shuffle(checker_odd)

    return checker_even + checker_odd


def reconstruct_image_checker_seeded(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
    random_seed: int,
) -> np.ndarray:
    """
    Rebuild the image using the assignment table, but place replacements
    in a deterministic checkerboard-aware order within each source color.
    """
    output_array = np.empty_like(image_array)
    rng = np.random.default_rng(random_seed)

    height, width, _ = image_array.shape

    positions_by_source: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for row_index in range(height):
        for column_index in range(width):
            source_rgb = tuple(int(channel) for channel in image_array[row_index, column_index])
            source_hex = rgb_to_hex(source_rgb)
            positions_by_source[source_hex].append((row_index, column_index))

    replacement_plan = build_source_replacement_plan(assignment_df)

    for source_hex, positions in positions_by_source.items():
        if source_hex not in replacement_plan:
            raise ValueError(f"No replacement plan available for source color: {source_hex}")

        ordered_positions = build_checker_position_order(
            positions=positions,
            rng=rng,
        )

        position_cursor = 0

        for bucket in replacement_plan[source_hex]:
            replacement_hex = bucket["ReplacementHex"]
            remaining_count = int(bucket["RemainingCount"])
            replacement_rgb = hex_to_rgb(replacement_hex)

            for _ in range(remaining_count):
                if position_cursor >= len(ordered_positions):
                    raise ValueError(
                        f"Ran out of positions while reconstructing source color {source_hex}"
                    )

                row_index, column_index = ordered_positions[position_cursor]
                output_array[row_index, column_index] = replacement_rgb
                position_cursor += 1

        if position_cursor != len(ordered_positions):
            raise ValueError(
                f"Not all positions were assigned for source color {source_hex}"
            )

    return output_array