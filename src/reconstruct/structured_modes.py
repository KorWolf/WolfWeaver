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


def build_block_position_order(
    positions: List[Tuple[int, int]],
    rng: np.random.Generator,
    block_size: int = 4,
) -> List[Tuple[int, int]]:
    """
    Return positions ordered by spatial blocks.

    Process:
    - group positions into blocks using integer division
    - shuffle the order of the blocks
    - shuffle positions inside each block
    - flatten everything into one ordered list

    Why this helps:
    - it keeps assignments more spatially local than full random placement
    - it often preserves image readability better for small palettes
    - it remains deterministic for a fixed seed
    """
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    blocks: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)

    for row_index, column_index in positions:
        block_key = (row_index // block_size, column_index // block_size)
        blocks[block_key].append((row_index, column_index))

    block_keys = list(blocks.keys())
    rng.shuffle(block_keys)

    ordered_positions: List[Tuple[int, int]] = []

    for block_key in block_keys:
        block_positions = list(blocks[block_key])
        rng.shuffle(block_positions)
        ordered_positions.extend(block_positions)

    return ordered_positions


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


def reconstruct_image_block_seeded(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
    random_seed: int,
    block_size: int = 4,
) -> np.ndarray:
    """
    Rebuild the image using the assignment table, but place replacements
    in a deterministic block-aware order within each source color.

    This preserves:
    - exact counts
    - deterministic output for the same seed
    - more local grouping than checker_seeded or full random

    The block size is fixed for now to keep the UI and config simple.
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

        ordered_positions = build_block_position_order(
            positions=positions,
            rng=rng,
            block_size=block_size,
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