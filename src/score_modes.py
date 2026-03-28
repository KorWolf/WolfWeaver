from typing import Callable


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
    rgb_difference = (
        abs(source_r - replacement_r)
        + abs(source_g - replacement_g)
        + abs(source_b - replacement_b)
    ) / 3.0

    saturation_difference = abs(source_s - replacement_s)
    lightness_difference = abs(source_l - replacement_l)

    return (
        (rgb_difference * 0.45)
        + (saturation_difference * 0.25)
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
    rgb_difference = (
        abs(source_r - replacement_r)
        + abs(source_g - replacement_g)
        + abs(source_b - replacement_b)
    ) / 3.0

    saturation_difference = abs(source_s - replacement_s)
    lightness_difference = abs(source_l - replacement_l)

    source_saturation_strength = source_s / 255.0
    accent_multiplier = 1.0 + (source_saturation_strength * 1.5)

    return (
        (rgb_difference * 0.40)
        + (lightness_difference * 0.25)
        + (saturation_difference * 0.20)
        + (saturation_difference * accent_multiplier * 0.15)
    )


SCORE_MODE_FUNCTIONS: dict[str, Callable[..., float]] = {
    "basic_rgb_sl": calculate_difference_score_basic,
    "weighted_rgb_sl": calculate_difference_score_weighted,
    "accent_aware": calculate_difference_score_accent_aware,
}


def get_score_mode_values() -> list[str]:
    return list(SCORE_MODE_FUNCTIONS.keys())


def get_score_function(score_mode: str) -> Callable[..., float]:
    if score_mode not in SCORE_MODE_FUNCTIONS:
        raise ValueError(
            f"Unsupported score_mode: {score_mode}. "
            f"Valid options: {get_score_mode_values()}"
        )
    return SCORE_MODE_FUNCTIONS[score_mode]