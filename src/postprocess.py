from collections import Counter
import re

import numpy as np


# ============================================================
# Palette / color helpers
# ============================================================

HEX_COLOR_PATTERN = re.compile(r"^#?([0-9A-Fa-f]{6})$")


def hex_to_rgb_array(hex_color: str) -> np.ndarray:
    """
    Convert a hex color string like '#AABBCC' into a numpy RGB array.
    """
    match = HEX_COLOR_PATTERN.fullmatch(hex_color.strip())
    if not match:
        raise ValueError(f"Invalid hex color: {hex_color}")

    hex_value = match.group(1)
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)

    return np.asarray([r, g, b], dtype=np.uint8)


def build_palette_array(palette_colors: list[str]) -> np.ndarray:
    """
    Convert palette hex colors into a (N, 3) uint8 numpy array.
    """
    if len(palette_colors) == 0:
        raise ValueError("palette_colors cannot be empty.")

    return np.asarray([hex_to_rgb_array(color) for color in palette_colors], dtype=np.uint8)


def calculate_rgb_distance(color_a: np.ndarray, color_b: np.ndarray) -> float:
    """
    Euclidean RGB distance between two colors.
    """
    color_a = np.asarray(color_a, dtype=np.float64)
    color_b = np.asarray(color_b, dtype=np.float64)
    return float(np.linalg.norm(color_a - color_b))


# ============================================================
# Index map helpers
# ============================================================

def build_palette_lookup(palette_array: np.ndarray) -> dict[tuple[int, int, int], int]:
    """
    Build a fast exact-RGB lookup from palette color -> palette index.
    """
    lookup: dict[tuple[int, int, int], int] = {}

    for index, rgb in enumerate(palette_array):
        lookup[(int(rgb[0]), int(rgb[1]), int(rgb[2]))] = index

    return lookup


def find_nearest_palette_index(rgb: np.ndarray, palette_array: np.ndarray) -> int:
    """
    Find the nearest palette index for a color when exact lookup fails.

    This is only used as a fallback. In normal operation, reconstructed
    pixels should already be exact palette colors.
    """
    distances = np.linalg.norm(
        palette_array.astype(np.float64) - rgb.astype(np.float64),
        axis=1,
    )
    return int(np.argmin(distances))


def build_palette_index_map(
    image_array: np.ndarray,
    palette_array: np.ndarray,
) -> np.ndarray:
    """
    Convert a reconstructed RGB image into a 2D map of palette indexes.
    """
    height, width = image_array.shape[:2]
    index_map = np.zeros((height, width), dtype=np.int32)

    palette_lookup = build_palette_lookup(palette_array)

    for y in range(height):
        for x in range(width):
            rgb = image_array[y, x]
            rgb_key = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            if rgb_key in palette_lookup:
                index_map[y, x] = palette_lookup[rgb_key]
            else:
                index_map[y, x] = find_nearest_palette_index(rgb, palette_array)

    return index_map


def palette_index_map_to_image(
    index_map: np.ndarray,
    palette_array: np.ndarray,
) -> np.ndarray:
    """
    Convert a 2D map of palette indexes back into an RGB image array.
    """
    return palette_array[index_map]


# ============================================================
# Edge detection helpers
# ============================================================

def build_edge_mask(
    source_image_array: np.ndarray,
    edge_threshold: float = 34.0,
) -> np.ndarray:
    """
    Build a simple edge mask from the source image.

    A pixel is treated as an edge pixel if its RGB change relative to
    nearby source pixels exceeds the threshold.
    """
    height, width = source_image_array.shape[:2]
    edge_mask = np.zeros((height, width), dtype=bool)

    source = source_image_array.astype(np.float64)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = source[y, x]

            left_distance = np.linalg.norm(center - source[y, x - 1])
            right_distance = np.linalg.norm(center - source[y, x + 1])
            up_distance = np.linalg.norm(center - source[y - 1, x])
            down_distance = np.linalg.norm(center - source[y + 1, x])

            edge_strength = max(
                left_distance,
                right_distance,
                up_distance,
                down_distance,
            )

            if edge_strength >= edge_threshold:
                edge_mask[y, x] = True

    return edge_mask


# ============================================================
# Neighbor-agreement refinement
# ============================================================

def refine_palette_index_map(
    source_image_array: np.ndarray,
    palette_index_map: np.ndarray,
    palette_array: np.ndarray,
    edge_mask: np.ndarray | None,
    passes: int = 1,
    majority_ratio: float = 0.625,
    min_majority_count: int = 5,
    candidate_slack: float = 10.0,
) -> np.ndarray:
    """
    Refine a palette index map using local neighbor agreement.

    Rules:
    - skip edge pixels when edge_mask is provided
    - inspect 8 neighboring palette indexes
    - if a strong local majority exists, consider switching
    - only switch when the majority color is not much worse than the
      current color relative to the source pixel
    """
    height, width = palette_index_map.shape
    working_map = palette_index_map.copy()

    for _ in range(passes):
        next_map = working_map.copy()

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if edge_mask is not None and edge_mask[y, x]:
                    continue

                current_index = int(working_map[y, x])

                neighbor_indexes = [
                    int(working_map[y - 1, x - 1]),
                    int(working_map[y - 1, x]),
                    int(working_map[y - 1, x + 1]),
                    int(working_map[y, x - 1]),
                    int(working_map[y, x + 1]),
                    int(working_map[y + 1, x - 1]),
                    int(working_map[y + 1, x]),
                    int(working_map[y + 1, x + 1]),
                ]

                counts = Counter(neighbor_indexes)
                winning_index, winning_count = counts.most_common(1)[0]

                if winning_index == current_index:
                    continue

                if winning_count < min_majority_count:
                    continue

                if (winning_count / 8.0) < majority_ratio:
                    continue

                source_rgb = source_image_array[y, x].astype(np.float64)
                current_rgb = palette_array[current_index].astype(np.float64)
                winning_rgb = palette_array[winning_index].astype(np.float64)

                current_distance = np.linalg.norm(source_rgb - current_rgb)
                winning_distance = np.linalg.norm(source_rgb - winning_rgb)

                # Only switch if the majority color is either better or
                # not significantly worse than the current match.
                if winning_distance <= (current_distance + candidate_slack):
                    next_map[y, x] = winning_index

        working_map = next_map

    return working_map


# ============================================================
# Public entry point
# ============================================================

def apply_postprocess(
    source_image_array: np.ndarray,
    reconstructed_image_array: np.ndarray,
    palette_colors: list[str],
    postprocess_mode: str = "none",
) -> np.ndarray:
    """
    Apply optional post-processing to the reconstructed image.

    Supported modes:
    - none
    - coherence_basic
    - coherence_edge_aware
    """
    if postprocess_mode == "none":
        return reconstructed_image_array

    if postprocess_mode not in {"coherence_basic", "coherence_edge_aware"}:
        raise ValueError(f"Unsupported postprocess_mode: {postprocess_mode}")

    palette_array = build_palette_array(palette_colors)
    palette_index_map = build_palette_index_map(
        image_array=reconstructed_image_array,
        palette_array=palette_array,
    )

    edge_mask = None
    if postprocess_mode == "coherence_edge_aware":
        edge_mask = build_edge_mask(
            source_image_array=source_image_array,
            edge_threshold=34.0,
        )

    refined_index_map = refine_palette_index_map(
        source_image_array=source_image_array,
        palette_index_map=palette_index_map,
        palette_array=palette_array,
        edge_mask=edge_mask,
        passes=1,
        majority_ratio=0.625,
        min_majority_count=5,
        candidate_slack=10.0,
    )

    return palette_index_map_to_image(
        index_map=refined_index_map,
        palette_array=palette_array,
    ).astype(np.uint8)