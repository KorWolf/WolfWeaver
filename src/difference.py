import pandas as pd
from src.score_modes import get_score_function


def build_difference_matrix(
    source_df: pd.DataFrame,
    palette_df: pd.DataFrame,
    score_mode: str = "basic_rgb_sl",
) -> pd.DataFrame:
    result_df = source_df.copy()
    score_function = get_score_function(score_mode)

    for _, palette_row in palette_df.iterrows():
        order = int(palette_row["Order"])
        replacement_hex = palette_row["Hex"]

        column_name = f"Score_{order:03d}_{replacement_hex}"

        result_df[column_name] = source_df.apply(
            lambda source_row: score_function(
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
    score_mode: str = "basic_rgb_sl",
) -> pd.DataFrame:
    score_function = get_score_function(score_mode)
    rows = []

    for _, source_row in source_df.iterrows():
        source_hex = source_row["Hex"]
        frequency = int(source_row["Frequency"])

        for _, palette_row in palette_df.iterrows():
            score = score_function(
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
            )

            rows.append(
                {
                    "SourceHex": source_hex,
                    "Frequency": frequency,
                    "ReplacementOrder": int(palette_row["Order"]),
                    "ReplacementHex": palette_row["Hex"],
                    "Score": round(score, 6),
                }
            )

    return pd.DataFrame(rows)


def export_difference_preview(
    difference_df: pd.DataFrame,
    output_path: str,
    num_rows: int = 200,
) -> None:
    difference_df.head(num_rows).to_csv(output_path, index=False)


def export_long_difference_preview(
    long_df: pd.DataFrame,
    output_path: str,
    num_rows: int = 1000,
) -> None:
    long_df.head(num_rows).to_csv(output_path, index=False)