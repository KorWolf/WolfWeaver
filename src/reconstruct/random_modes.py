from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.palette import hex_to_rgb

from .common import build_source_replacement_plan, rgb_to_hex


def reconstruct_image_random_seeded(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
    random_seed: int,
) -> np.ndarray:
    """
    Rebuild the image using the assignment table, but randomize placement
    among pixels of the same source color using a fixed random seed.

    This preserves:
    - exact counts
    - deterministic output for the same seed
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

        shuffled_positions = list(positions)
        rng.shuffle(shuffled_positions)

        position_cursor = 0

        for bucket in replacement_plan[source_hex]:
            replacement_hex = bucket["ReplacementHex"]
            remaining_count = int(bucket["RemainingCount"])
            replacement_rgb = hex_to_rgb(replacement_hex)

            for _ in range(remaining_count):
                if position_cursor >= len(shuffled_positions):
                    raise ValueError(
                        f"Ran out of positions while reconstructing source color {source_hex}"
                    )

                row_index, column_index = shuffled_positions[position_cursor]
                output_array[row_index, column_index] = replacement_rgb
                position_cursor += 1

        if position_cursor != len(shuffled_positions):
            raise ValueError(
                f"Not all positions were assigned for source color {source_hex}"
            )

    return output_array


def reconstruct_image_random_unseeded(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
) -> np.ndarray:
    """
    Same as random_seeded, but uses a fresh RNG each run so output varies.
    """
    output_array = np.empty_like(image_array)
    rng = np.random.default_rng()

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

        shuffled_positions = list(positions)
        rng.shuffle(shuffled_positions)

        position_cursor = 0

        for bucket in replacement_plan[source_hex]:
            replacement_hex = bucket["ReplacementHex"]
            remaining_count = int(bucket["RemainingCount"])
            replacement_rgb = hex_to_rgb(replacement_hex)

            for _ in range(remaining_count):
                if position_cursor >= len(shuffled_positions):
                    raise ValueError(
                        f"Ran out of positions while reconstructing source color {source_hex}"
                    )

                row_index, column_index = shuffled_positions[position_cursor]
                output_array[row_index, column_index] = replacement_rgb
                position_cursor += 1

        if position_cursor != len(shuffled_positions):
            raise ValueError(
                f"Not all positions were assigned for source color {source_hex}"
            )

    return output_array


def build_weighted_assignment_sequence(
    buckets: List[dict],
    rng: np.random.Generator,
) -> List[str]:
    """
    Build a sequence of replacement hex values that preserves exact counts while
    choosing among remaining buckets probabilistically.

    Lower scores get higher weight.
    """
    working_buckets = [
        {
            "ReplacementHex": bucket["ReplacementHex"],
            "RemainingCount": int(bucket["RemainingCount"]),
            "Score": float(bucket["Score"]),
        }
        for bucket in buckets
    ]

    total_count = sum(bucket["RemainingCount"] for bucket in working_buckets)
    sequence: List[str] = []

    for _ in range(total_count):
        available_buckets = [bucket for bucket in working_buckets if bucket["RemainingCount"] > 0]

        if not available_buckets:
            break

        weights = np.array(
            [1.0 / (bucket["Score"] + 1e-6) for bucket in available_buckets],
            dtype=np.float64,
        )
        probabilities = weights / weights.sum()

        chosen_index = int(rng.choice(len(available_buckets), p=probabilities))
        chosen_bucket = available_buckets[chosen_index]

        sequence.append(chosen_bucket["ReplacementHex"])
        chosen_bucket["RemainingCount"] -= 1

    return sequence


def reconstruct_image_weighted_random(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
    random_seed: int,
) -> np.ndarray:
    """
    Randomize placement within each source color, but choose the replacement
    sequence probabilistically so lower-score replacements are favored earlier.

    Exact counts are still preserved.
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

        shuffled_positions = list(positions)
        rng.shuffle(shuffled_positions)

        replacement_sequence = build_weighted_assignment_sequence(
            buckets=replacement_plan[source_hex],
            rng=rng,
        )

        if len(replacement_sequence) != len(shuffled_positions):
            raise ValueError(
                f"Weighted reconstruction mismatch for source color {source_hex}: "
                f"{len(replacement_sequence)} replacements vs {len(shuffled_positions)} positions."
            )

        for index, replacement_hex in enumerate(replacement_sequence):
            row_index, column_index = shuffled_positions[index]
            output_array[row_index, column_index] = hex_to_rgb(replacement_hex)

    return output_array