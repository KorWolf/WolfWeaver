from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.palette import hex_to_rgb

RGBTuple = Tuple[int, int, int]


def build_source_replacement_plan(assignment_df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Build a per-source-color replacement plan.

    For each source hex, keep an ordered list of replacement assignments:
    [
        {"ReplacementHex": "#00A94F", "RemainingCount": 120},
        {"ReplacementHex": "#009F4D", "RemainingCount": 15},
        ...
    ]
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
            }
        )

    return dict(plan)


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


def reconstruct_image_scanline(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
) -> np.ndarray:
    """
    Rebuild the image using the assignment table.

    Placement rule:
    - scan pixels in row-major order
    - for each source color, consume its replacement plan in order
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
    """
    output_array = np.empty_like(image_array)
    rng = np.random.default_rng(random_seed)

    height, width, _ = image_array.shape

    # Gather pixel positions for each source color
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

        # Shuffle positions deterministically using the provided seed
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

    raise ValueError(
        f"Unsupported reconstruction_mode: {reconstruction_mode}"
    )