import numpy as np
import pandas as pd

from src.palette import hex_to_rgb

from .common import build_source_replacement_plan, rgb_to_hex


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