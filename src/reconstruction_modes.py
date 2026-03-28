# ============================================================
# Reconstruction mode registry
# ============================================================
# This file is the single source of truth for reconstruction modes.
#
# Each mode includes:
# - value: the internal config / code value
# - label: what the user sees in the dropdown
# - description: help text shown in the UI
# - uses_seed: whether the random seed field should stay enabled
#
# When adding a new mode later, update this file first.
# ============================================================

RECONSTRUCTION_MODES = [
    {
        "value": "scanline",
        "label": "Scanline",
        "description": (
            "Scanline places replacement colors in the order the image is "
            "scanned from top-left to bottom-right. This gives a stable, "
            "structured result."
        ),
        "uses_seed": False,
    },
    {
        "value": "random_seeded",
        "label": "Random seeded",
        "description": (
            "Random seeded shuffles placement while staying repeatable. "
            "The same seed and settings should produce the same result."
        ),
        "uses_seed": True,
    },
    {
        "value": "checker_seeded",
        "label": "Checker seeded",
        "description": (
            "Checker seeded places replacements in a checkerboard-aware order "
            "within each source-color group, then uses the seed to keep that "
            "spread repeatable. This often reduces clumping while preserving "
            "exact counts."
        ),
        "uses_seed": True,
    },
    {
        "value": "block_seeded",
        "label": "Block seeded",
        "description": (
            "Block seeded places replacements in small shuffled spatial blocks "
            "within each source-color group. This often preserves local image "
            "structure better than broader random placement while staying repeatable."
        ),
        "uses_seed": True,
    },
    {
        "value": "random_unseeded",
        "label": "Random unseeded",
        "description": (
            "Random unseeded shuffles placement differently every run. "
            "Good for exploring new variations quickly."
        ),
        "uses_seed": False,
    },
    {
        "value": "weighted_random",
        "label": "Weighted random",
        "description": (
            "Weighted random uses randomness but favors lower-score assignments "
            "more strongly within each source-color group while preserving exact counts."
        ),
        "uses_seed": True,
    },
]


def get_reconstruction_mode_values() -> list[str]:
    """
    Return only the internal mode values, for validation and dispatch checks.
    """
    return [mode["value"] for mode in RECONSTRUCTION_MODES]