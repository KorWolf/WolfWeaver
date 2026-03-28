from typing import Callable
import colorsys


# ============================================================
# Color helpers
# ============================================================
# These helpers let score modes compare colors using:
# - weighted RGB difference
# - hue difference
# - saturation / lightness difference
# ============================================================

def rgb_to_hsv_components(r: int, g: int, b: int) -> tuple[float, float, float]:
    """
    Convert integer RGB values in the range 0-255 to HSV values
    in the range 0.0-1.0.
    """
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def calculate_weighted_rgb_difference(
    source_r: int,
    source_g: int,
    source_b: int,
    replacement_r: int,
    replacement_g: int,
    replacement_b: int,
) -> float:
    """
    Perceptually weighted RGB difference.

    Green differences matter more than blue differences here,
    which is closer to how people tend to notice color change.
    """
    return (
        abs(source_r - replacement_r) * 0.30
        + abs(source_g - replacement_g) * 0.59
        + abs(source_b - replacement_b) * 0.11
    )


def calculate_hue_distance(
    source_r: int,
    source_g: int,
    source_b: int,
    replacement_r: int,
    replacement_g: int,
    replacement_b: int,
) -> float:
    """
    Circular hue distance scaled into an RGB-like range.

    Hue wraps around, so the shortest path on the color wheel
    is used. The result is scaled so it can be combined more
    easily with the other score terms.
    """
    source_h, _, _ = rgb_to_hsv_components(source_r, source_g, source_b)
    replacement_h, _, _ = rgb_to_hsv_components(replacement_r, replacement_g, replacement_b)

    raw_distance = abs(source_h - replacement_h)
    circular_distance = min(raw_distance, 1.0 - raw_distance)

    return circular_distance * 255.0


def calculate_source_saturation_strength(
    source_r: int,
    source_g: int,
    source_b: int,
) -> float:
    """
    Return source saturation strength in the range 0.0-1.0.
    """
    _, source_s_hsv, _ = rgb_to_hsv_components(source_r, source_g, source_b)
    return float(source_s_hsv)


def calculate_replacement_saturation_strength(
    replacement_r: int,
    replacement_g: int,
    replacement_b: int,
) -> float:
    """
    Return replacement saturation strength in the range 0.0-1.0.
    """
    _, replacement_s_hsv, _ = rgb_to_hsv_components(replacement_r, replacement_g, replacement_b)
    return float(replacement_s_hsv)


def calculate_warm_bias(
    r: int,
    g: int,
    b: int,
) -> float:
    """
    Estimate how warm a color feels.

    Higher values mean the color leans more toward red/yellow warmth.
    This helps separate warm families like skin, cloth, gold, and hair.
    """
    return float((r * 0.60) + (g * 0.30) - (b * 0.20))


def calculate_warm_bias_difference(
    source_r: int,
    source_g: int,
    source_b: int,
    replacement_r: int,
    replacement_g: int,
    replacement_b: int,
) -> float:
    """
    Difference in warmth between source and replacement colors.
    """
    source_warm_bias = calculate_warm_bias(source_r, source_g, source_b)
    replacement_warm_bias = calculate_warm_bias(replacement_r, replacement_g, replacement_b)
    return abs(source_warm_bias - replacement_warm_bias)


# ============================================================
# Score modes
# ============================================================
# Each score mode returns a lower-is-better difference score.
# The assignment stage then uses those scores to choose matches.
# ============================================================

def calculate_difference_score_basic(
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
    Baseline score formula.

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


def calculate_difference_score_weighted(
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
    Weighted variant of the baseline score.

    This leans harder on weighted RGB difference while still keeping
    saturation and lightness as important secondary terms.
    """
    weighted_rgb_difference = calculate_weighted_rgb_difference(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )

    saturation_difference = abs(source_s - replacement_s)
    lightness_difference = abs(source_l - replacement_l)

    return (
        (weighted_rgb_difference * 0.52)
        + (saturation_difference * 0.18)
        + (lightness_difference * 0.30)
    )


def calculate_difference_score_accent_aware(
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
    Accent-aware scoring.

    This mode is tuned to help small vivid details survive better by:
    - using weighted RGB difference
    - using hue distance more strongly for vivid source colors
    - penalizing vivid -> dull matches more aggressively
    - preserving lightness structure so accents do not drift too far
    """
    weighted_rgb_difference = calculate_weighted_rgb_difference(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )

    saturation_difference = abs(source_s - replacement_s)
    lightness_difference = abs(source_l - replacement_l)
    hue_distance = calculate_hue_distance(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )

    source_sat_strength = calculate_source_saturation_strength(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
    )
    replacement_sat_strength = calculate_replacement_saturation_strength(
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )

    vivid_source = source_sat_strength >= 0.18
    vivid_replacement = replacement_sat_strength >= 0.12

    # Vivid colors should resist being matched to noticeably duller colors.
    dull_replacement_penalty = max(source_sat_strength - replacement_sat_strength, 0.0) * 95.0

    # Hue mismatches matter more when the source color is vivid.
    hue_weight_multiplier = 1.0 + (source_sat_strength * 2.4)

    # Add a stronger penalty when a vivid source color is also being
    # shifted far in hue to another vivid family.
    vivid_hue_penalty = 0.0
    if vivid_source and vivid_replacement and hue_distance > 14.0:
        vivid_hue_penalty = hue_distance * 0.70

    return (
        (weighted_rgb_difference * 0.24)
        + (lightness_difference * 0.18)
        + (saturation_difference * 0.12)
        + (hue_distance * 0.24 * hue_weight_multiplier)
        + (dull_replacement_penalty * 0.12)
        + (vivid_hue_penalty * 0.10)
    )


def calculate_difference_score_separation_aware(
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
    Separation-aware scoring.

    This mode is more aggressive about keeping nearby color families
    apart, especially warm hues that often drift into one another
    such as hair, skin, beige cloth, and gold accents.

    Compared with the earlier version, this iteration:
    - strengthens hue pressure further
    - adds warm-family separation
    - keeps lightness important so large regions do not collapse
    """
    weighted_rgb_difference = calculate_weighted_rgb_difference(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )

    saturation_difference = abs(source_s - replacement_s)
    lightness_difference = abs(source_l - replacement_l)
    hue_distance = calculate_hue_distance(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )
    warm_bias_difference = calculate_warm_bias_difference(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )

    source_sat_strength = calculate_source_saturation_strength(
        source_r=source_r,
        source_g=source_g,
        source_b=source_b,
    )
    replacement_sat_strength = calculate_replacement_saturation_strength(
        replacement_r=replacement_r,
        replacement_g=replacement_g,
        replacement_b=replacement_b,
    )

    vivid_source = source_sat_strength >= 0.16
    vivid_replacement = replacement_sat_strength >= 0.10
    source_is_warm = calculate_warm_bias(source_r, source_g, source_b) >= 80.0
    replacement_is_warm = calculate_warm_bias(replacement_r, replacement_g, replacement_b) >= 80.0

    dull_replacement_penalty = max(source_sat_strength - replacement_sat_strength, 0.0) * 105.0
    hue_weight_multiplier = 1.0 + (source_sat_strength * 2.6)

    # Push visibly different vivid families apart more strongly.
    vivid_family_penalty = 0.0
    if vivid_source and vivid_replacement and hue_distance > 12.0:
        vivid_family_penalty = hue_distance * 0.85

    # Warm families are where you were seeing drift, so add a specific
    # warm-bias separation term to reduce skin/hair/cloth blending.
    warm_family_penalty = 0.0
    if source_is_warm or replacement_is_warm:
        warm_family_penalty = warm_bias_difference * 0.35

    return (
        (weighted_rgb_difference * 0.20)
        + (lightness_difference * 0.18)
        + (saturation_difference * 0.10)
        + (hue_distance * 0.26 * hue_weight_multiplier)
        + (dull_replacement_penalty * 0.10)
        + (vivid_family_penalty * 0.10)
        + (warm_family_penalty * 0.06)
    )


# ============================================================
# Mode registry
# ============================================================
# This is the single source of truth for valid score mode names.
# ============================================================

SCORE_MODE_FUNCTIONS: dict[str, Callable[..., float]] = {
    "basic_rgb_sl": calculate_difference_score_basic,
    "weighted_rgb_sl": calculate_difference_score_weighted,
    "accent_aware": calculate_difference_score_accent_aware,
    "separation_aware": calculate_difference_score_separation_aware,
}


def get_score_mode_values() -> list[str]:
    """
    Return the supported score mode names.
    """
    return list(SCORE_MODE_FUNCTIONS.keys())


def get_score_function(score_mode: str) -> Callable[..., float]:
    """
    Return the scoring function for the requested mode.
    """
    if score_mode not in SCORE_MODE_FUNCTIONS:
        raise ValueError(
            f"Unsupported score_mode: {score_mode}. "
            f"Valid options: {get_score_mode_values()}"
        )

    return SCORE_MODE_FUNCTIONS[score_mode]