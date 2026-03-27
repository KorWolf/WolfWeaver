from typing import Dict, List

import pandas as pd


def build_assignment_table(
    source_df: pd.DataFrame,
    palette_df: pd.DataFrame,
    long_difference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the assignment table using a greedy, quota-constrained process.

    For each replacement color in palette order:
    1. Take all remaining source frequencies
    2. Sort remaining source colors by score for that replacement color
    3. Assign as much frequency as possible until the quota is filled
    4. Split source frequency if necessary

    Returns a DataFrame with columns:
    - SourceHex
    - AssignedCount
    - ReplacementOrder
    - ReplacementHex
    - Score
    """
    # Track how much of each source color remains unassigned.
    remaining_frequency: Dict[str, int] = {
        row["Hex"]: int(row["Frequency"])
        for _, row in source_df.iterrows()
    }

    assignment_rows: List[dict] = []

    for _, palette_row in palette_df.iterrows():
        replacement_order = int(palette_row["Order"])
        replacement_hex = str(palette_row["Hex"])
        quota_remaining = int(palette_row["Quota"])

        # Filter to the score rows for this replacement color.
        candidate_rows = long_difference_df[
            long_difference_df["ReplacementOrder"] == replacement_order
        ].copy()

        # Sort by:
        # 1. lowest score first
        # 2. source hex ascending for deterministic tie-breaking
        candidate_rows = candidate_rows.sort_values(
            by=["Score", "SourceHex"],
            ascending=[True, True],
        ).reset_index(drop=True)

        for _, candidate_row in candidate_rows.iterrows():
            if quota_remaining <= 0:
                break

            source_hex = str(candidate_row["SourceHex"])
            score = float(candidate_row["Score"])

            available = remaining_frequency.get(source_hex, 0)

            if available <= 0:
                continue

            assigned_count = min(available, quota_remaining)

            assignment_rows.append(
                {
                    "SourceHex": source_hex,
                    "AssignedCount": assigned_count,
                    "ReplacementOrder": replacement_order,
                    "ReplacementHex": replacement_hex,
                    "Score": round(score, 6),
                }
            )

            remaining_frequency[source_hex] -= assigned_count
            quota_remaining -= assigned_count

        if quota_remaining != 0:
            raise ValueError(
                f"Failed to fill quota for replacement color "
                f"{replacement_order} ({replacement_hex}). "
                f"Unfilled amount: {quota_remaining}"
            )

    # Final sanity check: all source frequency should be consumed.
    leftover_total = sum(remaining_frequency.values())
    if leftover_total != 0:
        raise ValueError(
            f"Assignment ended with leftover source frequency: {leftover_total}"
        )

    assignment_df = pd.DataFrame(assignment_rows)

    assignment_df = assignment_df.sort_values(
        by=["ReplacementOrder", "Score", "SourceHex"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return assignment_df


def build_assignment_summary(assignment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table showing how many pixels each replacement color received.
    """
    summary_df = (
        assignment_df.groupby(["ReplacementOrder", "ReplacementHex"], as_index=False)["AssignedCount"]
        .sum()
        .rename(columns={"AssignedCount": "TotalAssigned"})
        .sort_values(by=["ReplacementOrder"], ascending=True)
        .reset_index(drop=True)
    )

    return summary_df


def export_assignment_table(assignment_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the full assignment table to CSV.
    """
    assignment_df.to_csv(output_path, index=False)


def export_assignment_summary(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the assignment summary to CSV.
    """
    summary_df.to_csv(output_path, index=False)