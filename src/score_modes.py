from typing import Callable
import colorsys


# ============================================================
# Color helpers
# ============================================================
# These helpers let the score modes compare colors in a few
# different ways:
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
    which is generally closer to how people notice color change.
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
    easily with other score terms.
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

    This is used to make vivid source colors fight harder to
    stay vivid when matching.
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


# ============================================================
# Score modes
# ============================================================
# Each score mode returns a lower-is-better difference score.
# The pipeline then uses those scores during assignment.
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

    This leans harder on perceptually weighted RGB difference and
    slightly reduces the influence of the old saturation/lightness
    terms compared with the original baseline.
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
        (weighted_rgb_difference * 0.50)
        + (saturation_difference * 0.20)
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

    This keeps the basic RGB / saturation / lightness idea, but:
    - uses weighted RGB difference
    - adds hue separation
    - makes vivid source colors fight harder to stay vivid
    - penalizes matching a vivid source color to a dull replacement
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

    # More saturated source colors should resist being mapped to dull colors.
    dull_replacement_penalty = max(source_sat_strength - replacement_sat_strength, 0.0) * 60.0

    # Hue differences matter more when the source color is vivid.
    hue_weight_multiplier = 1.0 + (source_sat_strength * 1.8)

    return (
        (weighted_rgb_difference * 0.30)
        + (lightness_difference * 0.20)
        + (saturation_difference * 0.15)
        + (hue_distance * 0.20 * hue_weight_multiplier)
        + (dull_replacement_penalty * 0.15)
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

    It does this by:
    - using weighted RGB difference
    - using hue distance more strongly
    - increasing hue pressure when the source is vivid
    - applying an extra family-separation penalty for visibly
      different saturated colors
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

    family_separation_penalty = 0.0
    if vivid_source and vivid_replacement and hue_distance > 18.0:
        family_separation_penalty = hue_distance * 0.65

    dull_replacement_penalty = max(source_sat_strength - replacement_sat_strength, 0.0) * 75.0
    hue_weight_multiplier = 1.0 + (source_sat_strength * 2.0)

    return (
        (weighted_rgb_difference * 0.24)
        + (lightness_difference * 0.18)
        + (saturation_difference * 0.12)
        + (hue_distance * 0.24 * hue_weight_multiplier)
        + (dull_replacement_penalty * 0.10)
        + (family_separation_penalty * 0.12)
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