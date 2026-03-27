import pandas as pd
import numpy as np


def calculate_difference_score(
    source_r: int,
    source_g: int,
    source_b: int,
    source_s: int,
    source_l: float,
    replacement_r: int,
    replacement_g: int,
    replacement_b: int,
    replacement_s: int,
    replacement_l: float,
) -> float:
    """
    Calculate the color difference score using the same method as the Excel process.

    Formula:
    ((abs(R1 - R2) + abs(G1 - G2) + abs(B1 - B2)) / 3)
    + abs(S1 - S2)
    + abs(L1 - L2)
    """
    rgb_difference = (
        abs(source_r - replacement_r)
        + abs(source_g - replacement_g)
        + abs(source_b - replacement_b)
    ) / 3.0

    saturation_difference = abs(source_s - replacement_s)
    lightness_difference = abs(source_l - replacement_l)

    return rgb_difference + saturation_difference + lightness_difference


def build_difference_matrix(
    source_df: pd.DataFrame,
    palette_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a DataFrame where:
    - each row is a source color
    - each replacement color gets its own score column

    Output includes:
    - source color metadata
    - one score column per replacement palette entry
    """
    result_df = source_df.copy()

    for _, palette_row in palette_df.iterrows():
        order = int(palette_row["Order"])
        replacement_hex = palette_row["Hex"]

        column_name = f"Score_{order:03d}_{replacement_hex}"

        result_df[column_name] = source_df.apply(
            lambda source_row: calculate_difference_score(
                source_r=int(source_row["R"]),
                source_g=int(source_row["G"]),
                source_b=int(source_row["B"]),
                source_s=int(source_row["S"]),
                source_l=float(source_row["L"]),
                replacement_r=int(palette_row["R"]),
                replacement_g=int(palette_row["G"]),
                replacement_b=int(palette_row["B"]),
                replacement_s=int(palette_row["S"]),
                replacement_l=float(palette_row["L"]),
            ),
            axis=1,
        )

    return result_df


def build_long_difference_table(
    source_df: pd.DataFrame,
    palette_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a long-form difference table with one row per source/replacement pair.

    Output columns:
    - SourceHex
    - Frequency
    - ReplacementOrder
    - ReplacementHex
    - Score

    This format is useful later for assignment logic and debugging.
    """
    rows = []

    for _, source_row in source_df.iterrows():
        source_hex = source_row["Hex"]
        frequency = int(source_row["Frequency"])
        source_r = int(source_row["R"])
        source_g = int(source_row["G"])
        source_b = int(source_row["B"])
        source_s = int(source_row["S"])
        source_l = float(source_row["L"])

        for _, palette_row in palette_df.iterrows():
            replacement_order = int(palette_row["Order"])
            replacement_hex = palette_row["Hex"]

            score = calculate_difference_score(
                source_r=source_r,
                source_g=source_g,
                source_b=source_b,
                source_s=source_s,
                source_l=source_l,
                replacement_r=int(palette_row["R"]),
                replacement_g=int(palette_row["G"]),
                replacement_b=int(palette_row["B"]),
                replacement_s=int(palette_row["S"]),
                replacement_l=float(palette_row["L"]),
            )

            rows.append(
                {
                    "SourceHex": source_hex,
                    "Frequency": frequency,
                    "ReplacementOrder": replacement_order,
                    "ReplacementHex": replacement_hex,
                    "Score": round(score, 6),
                }
            )

    return pd.DataFrame(rows)


def export_difference_preview(
    difference_df: pd.DataFrame,
    output_path: str,
    num_rows: int = 200,
) -> None:
    """
    Export only the first N rows of the wide difference matrix for inspection.
    """
    preview_df = difference_df.head(num_rows)
    preview_df.to_csv(output_path, index=False)


def export_long_difference_preview(
    long_df: pd.DataFrame,
    output_path: str,
    num_rows: int = 1000,
) -> None:
    """
    Export only the first N rows of the long-form difference table for inspection.
    """
    preview_df = long_df.head(num_rows)
    preview_df.to_csv(output_path, index=False)