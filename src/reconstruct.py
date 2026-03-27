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

    Order is determined by:
    - ReplacementOrder ascending
    - Score ascending
    - ReplacementHex ascending
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


def reconstruct_image_from_assignments(
    image_array: np.ndarray,
    assignment_df: pd.DataFrame,
) -> np.ndarray:
    """
    Rebuild the image using the assignment table.

    Deterministic placement rule:
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

    # Final sanity check: all planned counts should be fully consumed.
    leftover = 0
    for buckets in replacement_plan.values():
        for bucket in buckets:
            leftover += int(bucket["RemainingCount"])

    if leftover != 0:
        raise ValueError(f"Reconstruction ended with leftover planned assignments: {leftover}")

    return output_array


def save_image_array(image_array: np.ndarray, output_path: Path) -> None:
    """
    Save an RGB NumPy array as a PNG image.
    """
    image = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
    image.save(output_path)