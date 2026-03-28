from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# Type alias used throughout the reconstruction modules.
RGBTuple = Tuple[int, int, int]


def build_source_replacement_plan(assignment_df: pd.DataFrame) -> Dict[str, List[dict]]:
    """
    Build a per-source-color replacement plan.

    For each source hex, keep an ordered list of replacement assignments:
    [
        {"ReplacementHex": "#00A94F", "RemainingCount": 120, "Score": 12.5},
        {"ReplacementHex": "#009F4D", "RemainingCount": 15, "Score": 14.0},
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
                "Score": float(row["Score"]),
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