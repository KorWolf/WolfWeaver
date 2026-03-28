from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.palette import hex_to_rgb

# Type alias used throughout this file for readability.
# An RGB tuple is always ordered as (R, G, B).
RGBTuple = Tuple[int, int, int]


# ============================================================
# Assignment-plan helpers
# ============================================================
# These helpers convert the assignment table into a structure
# that is easier for reconstruction modes to consume.
#
# The assignment table says:
# "for source color X, assign N pixels to replacement color Y"
#
# Reconstruction modes then decide WHERE those assigned pixels
# should be placed in the output image.
# ============================================================

def build_source_replacement_plan(assignment_df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Build a per-source-color replacement plan.

    For each source hex, keep an ordered list of replacement assignments:
    [
        {"ReplacementHex": "#00A94F", "RemainingCount": 120, "Score": 12.5},
        {"ReplacementHex": "#009F4D", "RemainingCount": 15, "Score": 14.0},
        ...
    ]

    Why this exists:
    - The assignment table is a flat DataFrame.
    - Reconstruction needs a grouped structure keyed by source color.
    - Each source color can map to multiple replacement colors.
    """
    plan: Dict[str, List[dict]] = defaultdict(list)

    sorted_df = assignment_df.sort_values(
        by=["SourceHex", "ReplacementOrder", "Score", "ReplacementHex"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    for _, row in sorted_df.iterrows():
        plan[str(row["SourceHex"])].append(
            {
                "ReplacementHex": str(row["ReplacementHex"]),
                "RemainingCount": int(row["AssignedCount"]),
                "Score": float(row["Score"]),
            }
        )

    return dict(plan)


# ============================================================
# Basic image / color helpers
# ============================================================
# Small utility helpers used by multiple reconstruction modes.
# ============================================================

def rgb_to_hex(rgb: RGBTuple) -> str:
    """
    Convert an RGB tuple into uppercase hex string.
    """
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def save_image_array(image_array: np.ndarray, output_path: Path) -> None:
    """
    Save an RGB NumPy array as a PNG image.
    """
    image = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
    image.save(output_path)


# ============================================================
# Position-order helpers
# ============================================================
# These helpers decide the order in which positions are consumed
# for a given source color.
#
# Different reconstruction modes mostly differ in how they order
# or choose these positions.
# ============================================================

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

    Why this helps:
    - It spreads assignments across the image more evenly than
      a plain row-by-row or fully local sequence.
    - It is still deterministic when the same seed is used.
    - It often reduces visible clumping.

    Note:
    This is "checker traversal", not "forced checker patterning".
    It changes the order positions are filled, but does not force
    alternating colors everywhere.
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


# ============================================================
# Reconstruction modes
# ============================================================
# Each mode takes:
# - the original image array
# - the assignment table
# - sometimes a random seed
#
# Each mode must:
# - preserve exact assigned counts
# - fill every pixel exactly once
# - return a full output image array
# ============================================================

def reconstruct_image_scanline(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
) -> np.ndarray:
    """
    Rebuild the image using the assignment table.

    Placement rule:
    - scan pixels in row-major order
    - for each source color, consume its replacement plan in order

    This is the most structured and deterministic mode.
    It can, however, create visible directional artifacts because
    placement always follows the same top-left to bottom-right flow.
    """
    output_array = np.empty_like(image_array)
    replacement_plan = build_source_replacement_plan(assignment_df)

    height, width, _ = image_array.shape

    for row_index in range(height):
        for column_index in range(width):
            source_rgb = tuple(int(channel) for channel in image_array[row_index, column_index])
            source_hex = rgb_to_hex(source_rgb)

            if source_hex not in replacement_plan or len(replacement_plan[source_hex]) == 0:
                raise ValueError(f"No replacement plan available for source color: {source_hex}")

            current_bucket = replacement_plan[source_hex][0]
            replacement_rgb = hex_to_rgb(current_bucket["ReplacementHex"])

            output_array[row_index, column_index] = replacement_rgb

            current_bucket["RemainingCount"] -= 1

            if current_bucket["RemainingCount"] < 0:
                raise ValueError(
                    f"Replacement plan over-consumed for source color {source_hex}"
                )

            if current_bucket["RemainingCount"] == 0:
                replacement_plan[source_hex].pop(0)

    leftover = 0
    for buckets in replacement_plan.values():
        for bucket in buckets:
            leftover += int(bucket["RemainingCount"])

    if leftover != 0:
        raise ValueError(f"Reconstruction ended with leftover planned assignments: {leftover}")

    return output_array


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

    This is useful when you want variation without losing reproducibility.
    """
    output_array = np.empty_like(image_array)
    rng = np.random.default_rng(random_seed)

    height, width, _ = image_array.shape

    # Group all pixel positions by their original source color.
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

        # Consume replacement buckets in order, placing each replacement
        # into the next shuffled position.
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


def reconstruct_image_checker_seeded(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
    random_seed: int,
) -> np.ndarray:
    """
    Rebuild the image using the assignment table, but place replacements
    in a deterministic checkerboard-aware order within each source color.

    This preserves:
    - exact counts
    - deterministic output for the same seed
    - broader spatial spread than plain scanline

    Process:
    - collect all positions for each source color
    - split positions into checker classes
    - shuffle within each checker class using the seed
    - fill replacements in that structured order

    Why this mode exists:
    - fully random placement can become noisy
    - scanline can create directional artifacts
    - checker_seeded sits in between those two behaviors
    """
    output_array = np.empty_like(image_array)
    rng = np.random.default_rng(random_seed)

    height, width, _ = image_array.shape

    # Group all positions by source color, same as the other grouped modes.
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

        # Instead of fully shuffling all positions together, build a
        # checker-aware position order.
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


# ============================================================
# Weighted-random helpers and mode
# ============================================================
# This mode still preserves exact counts, but changes the order of
# replacement usage using probabilities based on score.
# Lower score = more likely to be placed earlier.
# ============================================================

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

        # Inverse-score weighting:
        # lower score => larger weight => higher probability of being chosen.
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


# ============================================================
# Dispatcher
# ============================================================
# This is the single entry point the rest of the pipeline uses.
# The pipeline passes in the selected reconstruction_mode, and
# this function routes to the correct implementation.
# ============================================================

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
        return reconstruct_image_scanline(image_array=image_array, assignment_df=assignment_df)

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

    raise ValueError(
        f"Unsupported reconstruction_mode: {reconstruction_mode}"
    )