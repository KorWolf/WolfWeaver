from collections import Counter
from pathlib import Path
import colorsys

import numpy as np
from sklearn.cluster import KMeans

from src.color_stats import build_color_stats_dataframe, extract_color_frequencies
from src.image_io import load_image


# ============================================================
# Basic color helpers
# ============================================================

def rgb_array_to_hex(rgb: np.ndarray) -> str:
    """
    Convert a length-3 RGB array into a hex color string.
    """
    r, g, b = [int(np.clip(round(channel), 0, 255)) for channel in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def calculate_rgb_saturation(rgb: np.ndarray) -> float:
    """
    Simple RGB saturation measure:
    max channel - min channel
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    return float(np.max(rgb) - np.min(rgb))


def calculate_rgb_lightness(rgb: np.ndarray) -> float:
    """
    Simple RGB lightness measure:
    average of the three channels
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    return float(np.mean(rgb))


def calculate_rgb_distance(color_a: np.ndarray, color_b: np.ndarray) -> float:
    """
    Euclidean distance between two RGB colors.
    """
    color_a = np.asarray(color_a, dtype=np.float64)
    color_b = np.asarray(color_b, dtype=np.float64)
    return float(np.linalg.norm(color_a - color_b))


def calculate_hue_bucket(rgb: np.ndarray, bucket_count: int = 8) -> int | None:
    """
    Convert an RGB color into a coarse hue bucket.

    Returns None for low-saturation colors because they do not belong
    strongly to a hue family.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r = float(rgb[0]) / 255.0
    g = float(rgb[1]) / 255.0
    b = float(rgb[2]) / 255.0

    hue, saturation, _ = colorsys.rgb_to_hsv(r, g, b)

    # Low-saturation colors are treated as neutral rather than forced
    # into an arbitrary hue family.
    if saturation < 0.12:
        return None

    return int(hue * bucket_count) % bucket_count


def calculate_warmth(rgb: np.ndarray) -> float:
    """
    Estimate how warm a color feels.

    Higher values mean the color leans more toward warm red/yellow
    families, which helps some selection modes avoid over-filling
    with too many warm variants.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb
    return float((r * 0.60) + (g * 0.30) - (b * 0.20))


# ============================================================
# Pixel preparation helpers
# ============================================================

def sample_image_pixels(
    image_array: np.ndarray,
    max_samples: int = 50000,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Flatten image pixels to shape (N, 3) and randomly sample if needed.

    This keeps clustering fast on larger images.
    """
    pixels = image_array.reshape(-1, 3).astype(np.float64)

    if len(pixels) <= max_samples:
        return pixels

    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(len(pixels), size=max_samples, replace=False)
    return pixels[sample_indices]


def cluster_pixels_kmeans(
    pixels: np.ndarray,
    cluster_count: int,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster RGB pixels using KMeans.

    Returns:
    - centroids: shape (cluster_count, 3)
    - labels: shape (num_pixels,)
    """
    if cluster_count < 1:
        raise ValueError("cluster_count must be at least 1.")

    if len(pixels) < cluster_count:
        cluster_count = len(pixels)

    model = KMeans(
        n_clusters=cluster_count,
        random_state=random_seed,
        n_init=10,
    )

    labels = model.fit_predict(pixels)
    centroids = model.cluster_centers_

    return centroids, labels


# ============================================================
# Cluster representative helpers
# ============================================================

def get_nearest_real_rgb_for_cluster(
    cluster_pixels: np.ndarray,
    centroid_rgb: np.ndarray,
) -> np.ndarray:
    """
    Choose the actual cluster pixel closest to the centroid.

    This avoids inventing new colors that were not in the image.
    """
    distances = np.linalg.norm(cluster_pixels - centroid_rgb, axis=1)
    nearest_index = int(np.argmin(distances))
    return cluster_pixels[nearest_index]


def get_most_frequent_real_rgb_for_cluster(
    cluster_pixels: np.ndarray,
) -> np.ndarray:
    """
    Choose the most frequent exact RGB color inside the cluster.

    Ties are broken by lexicographic RGB ordering for stability.
    """
    rgb_tuples = [tuple(int(channel) for channel in pixel) for pixel in cluster_pixels]
    frequency_counter = Counter(rgb_tuples)

    most_common_items = frequency_counter.most_common()
    highest_count = most_common_items[0][1]

    tied_colors = [rgb for rgb, count in most_common_items if count == highest_count]
    chosen_rgb = min(tied_colors)

    return np.asarray(chosen_rgb, dtype=np.float64)


def build_cluster_records(
    centroids: np.ndarray,
    labels: np.ndarray,
    sampled_pixels: np.ndarray,
    representative_mode: str = "nearest_real",
) -> list[dict]:
    """
    Build metadata records for each cluster.

    representative_mode:
    - nearest_real: use the real pixel nearest to the centroid
    - most_frequent_real: use the most common exact color in the cluster
    """
    if representative_mode not in {"nearest_real", "most_frequent_real"}:
        raise ValueError(
            f"Unsupported representative_mode: {representative_mode}"
        )

    total_pixels = len(sampled_pixels)
    global_mean = np.mean(sampled_pixels, axis=0)

    records: list[dict] = []

    for cluster_index, centroid in enumerate(centroids):
        cluster_mask = labels == cluster_index
        count = int(np.sum(cluster_mask))

        if count <= 0:
            continue

        cluster_pixels = sampled_pixels[cluster_mask]

        if representative_mode == "nearest_real":
            representative_rgb = get_nearest_real_rgb_for_cluster(
                cluster_pixels=cluster_pixels,
                centroid_rgb=centroid,
            )
        else:
            representative_rgb = get_most_frequent_real_rgb_for_cluster(
                cluster_pixels=cluster_pixels,
            )

        representative_rgb = np.asarray(representative_rgb, dtype=np.float64)

        saturation = calculate_rgb_saturation(representative_rgb)
        lightness = calculate_rgb_lightness(representative_rgb)
        distance_from_global_mean = calculate_rgb_distance(representative_rgb, global_mean)
        hue_bucket = calculate_hue_bucket(representative_rgb)
        warmth = calculate_warmth(representative_rgb)

        records.append(
            {
                "rgb": representative_rgb,
                "hex": rgb_array_to_hex(representative_rgb),
                "count": count,
                "size_ratio": count / total_pixels,
                "saturation": saturation,
                "lightness": lightness,
                "distance_from_global_mean": distance_from_global_mean,
                "hue_bucket": hue_bucket,
                "warmth": warmth,
            }
        )

    return records


# ============================================================
# Cluster scoring helpers
# ============================================================

def normalize_values(values: list[float]) -> list[float]:
    """
    Normalize a list of floats to the range [0, 1].

    If all values are equal, return 0.0 for all entries.
    """
    if len(values) == 0:
        return []

    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return [0.0 for _ in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def score_cluster_records(cluster_records: list[dict]) -> list[dict]:
    """
    Add a balanced priority score to each cluster record.

    The score favors:
    - larger regions
    - more saturated colors
    - colors that stand out from the overall image average
    """
    saturations = [record["saturation"] for record in cluster_records]
    distances = [record["distance_from_global_mean"] for record in cluster_records]

    normalized_saturations = normalize_values(saturations)
    normalized_distances = normalize_values(distances)

    scored_records: list[dict] = []

    for index, record in enumerate(cluster_records):
        score = (
            (record["size_ratio"] * 0.65)
            + (normalized_saturations[index] * 0.20)
            + (normalized_distances[index] * 0.15)
        )

        scored_record = dict(record)
        scored_record["priority_score"] = float(score)
        scored_records.append(scored_record)

    scored_records.sort(
        key=lambda record: (
            -record["priority_score"],
            -record["size_ratio"],
            -record["saturation"],
            record["hex"],
        )
    )

    return scored_records


# ============================================================
# Final palette selection helpers
# ============================================================

def can_add_color(
    candidate_rgb: np.ndarray,
    selected_rgbs: list[np.ndarray],
    min_color_distance: float,
) -> bool:
    """
    Return True if the candidate color is far enough away from all selected colors.
    """
    for selected_rgb in selected_rgbs:
        if calculate_rgb_distance(candidate_rgb, selected_rgb) < min_color_distance:
            return False

    return True


def select_clustered_palette_colors(
    cluster_records: list[dict],
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 28.0,
) -> list[str]:
    """
    Select the final palette from scored cluster records.

    Strategy:
    - optionally reserve darkest and lightest colors first
    - fill remaining slots by priority score
    - enforce minimum color distance where possible
    - relax the distance rule in a fallback pass if needed
    """
    if color_count < 1:
        raise ValueError("color_count must be at least 1.")

    if len(cluster_records) == 0:
        raise ValueError("No cluster records were available for palette selection.")

    selected_records: list[dict] = []
    selected_rgbs: list[np.ndarray] = []

    def try_add_record(record: dict, distance_threshold: float) -> bool:
        rgb = record["rgb"]
        if can_add_color(rgb, selected_rgbs, distance_threshold):
            selected_records.append(record)
            selected_rgbs.append(rgb)
            return True
        return False

    if preserve_darkest:
        darkest_record = min(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_record(darkest_record, 0.0)

    if preserve_lightest and len(selected_records) < color_count:
        lightest_record = max(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        if lightest_record["hex"] not in {record["hex"] for record in selected_records}:
            try_add_record(lightest_record, min_color_distance)

    for record in cluster_records:
        if len(selected_records) >= color_count:
            break

        if record["hex"] in {selected["hex"] for selected in selected_records}:
            continue

        try_add_record(record, min_color_distance)

    if len(selected_records) < color_count:
        relaxed_threshold = max(min_color_distance * 0.5, 10.0)

        for record in cluster_records:
            if len(selected_records) >= color_count:
                break

            if record["hex"] in {selected["hex"] for selected in selected_records}:
                continue

            try_add_record(record, relaxed_threshold)

    if len(selected_records) < color_count:
        for record in cluster_records:
            if len(selected_records) >= color_count:
                break

            if record["hex"] in {selected["hex"] for selected in selected_records}:
                continue

            selected_records.append(record)
            selected_rgbs.append(record["rgb"])

    return [record["hex"] for record in selected_records[:color_count]]


def select_clustered_palette_colors_balanced(
    cluster_records: list[dict],
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 28.0,
) -> list[str]:
    """
    Select the final palette from scored cluster records with mild
    diversity pressure.

    This mode is intended to sit between:
    - standard: mostly dominance-driven
    - diverse: strongly spread across hue families

    Balanced mode:
    - still respects strong clusters
    - gently rewards new hue buckets
    - lightly discourages over-filling with warm variants
    - avoids forcing diversity too aggressively on simpler images
    """
    if color_count < 1:
        raise ValueError("color_count must be at least 1.")

    if len(cluster_records) == 0:
        raise ValueError("No cluster records were available for palette selection.")

    selected_records: list[dict] = []
    selected_rgbs: list[np.ndarray] = []
    selected_hexes: set[str] = set()

    def add_record(record: dict) -> None:
        selected_records.append(record)
        selected_rgbs.append(record["rgb"])
        selected_hexes.add(record["hex"])

    def try_add_anchor(record: dict, distance_threshold: float) -> bool:
        if record["hex"] in selected_hexes:
            return False

        if can_add_color(record["rgb"], selected_rgbs, distance_threshold):
            add_record(record)
            return True

        return False

    if preserve_darkest:
        darkest_record = min(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(darkest_record, 0.0)

    if preserve_lightest and len(selected_records) < color_count:
        lightest_record = max(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(lightest_record, min_color_distance)

    while len(selected_records) < color_count:
        best_candidate = None
        best_candidate_score = None

        used_hue_buckets = {
            record["hue_bucket"]
            for record in selected_records
            if record["hue_bucket"] is not None
        }

        selected_warm_count = sum(1 for record in selected_records if record["warmth"] >= 80.0)
        selected_total_count = len(selected_records)

        for record in cluster_records:
            if record["hex"] in selected_hexes:
                continue

            if not can_add_color(record["rgb"], selected_rgbs, min_color_distance):
                continue

            candidate_score = float(record["priority_score"])

            # Mild reward for bringing in a new hue family.
            if record["hue_bucket"] is not None and record["hue_bucket"] not in used_hue_buckets:
                candidate_score += 0.09

            # Only gently discourage additional warm colors if the current
            # selection is already strongly warm-dominated.
            if selected_total_count > 0:
                warm_ratio = selected_warm_count / selected_total_count

                if warm_ratio >= 0.65 and record["warmth"] >= 80.0:
                    candidate_score -= 0.04

                if warm_ratio >= 0.65 and record["warmth"] < 80.0:
                    candidate_score += 0.05

            # Slight reward for usable saturation so the palette can still
            # preserve some visual richness.
            if record["saturation"] >= 35.0 and record["size_ratio"] >= 0.01:
                candidate_score += 0.02

            if best_candidate is None or candidate_score > best_candidate_score:
                best_candidate = record
                best_candidate_score = candidate_score

        if best_candidate is not None:
            add_record(best_candidate)
            continue

        relaxed_threshold = max(min_color_distance * 0.5, 10.0)

        for record in cluster_records:
            if record["hex"] in selected_hexes:
                continue

            if can_add_color(record["rgb"], selected_rgbs, relaxed_threshold):
                add_record(record)
                break
        else:
            for record in cluster_records:
                if record["hex"] in selected_hexes:
                    continue

                add_record(record)
                break
            else:
                break

    return [record["hex"] for record in selected_records[:color_count]]


def select_clustered_palette_colors_diverse(
    cluster_records: list[dict],
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 28.0,
) -> list[str]:
    """
    Select the final palette from scored cluster records, with stronger
    diversity pressure than the standard selection mode.

    Goals:
    - still respect strong clusters
    - still keep dark/light anchors
    - avoid spending too many slots on one hue family
    - avoid over-filling with very similar warm tones
    """
    if color_count < 1:
        raise ValueError("color_count must be at least 1.")

    if len(cluster_records) == 0:
        raise ValueError("No cluster records were available for palette selection.")

    selected_records: list[dict] = []
    selected_rgbs: list[np.ndarray] = []
    selected_hexes: set[str] = set()

    def add_record(record: dict) -> None:
        selected_records.append(record)
        selected_rgbs.append(record["rgb"])
        selected_hexes.add(record["hex"])

    def try_add_anchor(record: dict, distance_threshold: float) -> bool:
        if record["hex"] in selected_hexes:
            return False

        if can_add_color(record["rgb"], selected_rgbs, distance_threshold):
            add_record(record)
            return True

        return False

    # Preserve strongest structural anchors first.
    if preserve_darkest:
        darkest_record = min(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(darkest_record, 0.0)

    if preserve_lightest and len(selected_records) < color_count:
        lightest_record = max(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(lightest_record, min_color_distance)

    # Diversity-aware greedy selection.
    while len(selected_records) < color_count:
        best_candidate = None
        best_candidate_score = None

        # Track currently used hue buckets and warm-family density.
        used_hue_buckets = {
            record["hue_bucket"]
            for record in selected_records
            if record["hue_bucket"] is not None
        }

        selected_warm_count = sum(1 for record in selected_records if record["warmth"] >= 80.0)
        selected_total_count = len(selected_records)

        for record in cluster_records:
            if record["hex"] in selected_hexes:
                continue

            # Keep basic distinctness.
            if not can_add_color(record["rgb"], selected_rgbs, min_color_distance):
                continue

            candidate_score = float(record["priority_score"])

            # Reward new hue-family coverage.
            if record["hue_bucket"] is not None and record["hue_bucket"] not in used_hue_buckets:
                candidate_score += 0.18

            # Reward cooler or underrepresented families when warm colors
            # are already dominating the selection.
            if selected_total_count > 0:
                warm_ratio = selected_warm_count / selected_total_count

                if warm_ratio >= 0.50 and record["warmth"] < 80.0:
                    candidate_score += 0.12

                if warm_ratio >= 0.60 and record["warmth"] >= 80.0:
                    candidate_score -= 0.10

            # Slightly reward saturated colors if they are not extremely tiny.
            if record["saturation"] >= 40.0 and record["size_ratio"] >= 0.01:
                candidate_score += 0.04

            if best_candidate is None or candidate_score > best_candidate_score:
                best_candidate = record
                best_candidate_score = candidate_score

        if best_candidate is not None:
            add_record(best_candidate)
            continue

        # Fallback pass: relax the distance rule if diversity pressure got too strict.
        relaxed_threshold = max(min_color_distance * 0.5, 10.0)

        for record in cluster_records:
            if record["hex"] in selected_hexes:
                continue

            if can_add_color(record["rgb"], selected_rgbs, relaxed_threshold):
                add_record(record)
                break
        else:
            for record in cluster_records:
                if record["hex"] in selected_hexes:
                    continue

                add_record(record)
                break
            else:
                break

    return [record["hex"] for record in selected_records[:color_count]]


def select_clustered_palette_colors_low_variety(
    cluster_records: list[dict],
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 32.0,
) -> list[str]:
    """
    Select the final palette from scored cluster records for low-variety images.

    This mode is designed to:
    - prefer dominant exact colors
    - preserve clarity
    - avoid forcing broad hue-family diversity
    - avoid spending slots on tiny outlier colors
    """
    if color_count < 1:
        raise ValueError("color_count must be at least 1.")

    if len(cluster_records) == 0:
        raise ValueError("No cluster records were available for palette selection.")

    selected_records: list[dict] = []
    selected_rgbs: list[np.ndarray] = []
    selected_hexes: set[str] = set()

    def add_record(record: dict) -> None:
        selected_records.append(record)
        selected_rgbs.append(record["rgb"])
        selected_hexes.add(record["hex"])

    def try_add_anchor(record: dict, distance_threshold: float) -> bool:
        if record["hex"] in selected_hexes:
            return False

        if can_add_color(record["rgb"], selected_rgbs, distance_threshold):
            add_record(record)
            return True

        return False

    if preserve_darkest:
        darkest_record = min(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(darkest_record, 0.0)

    if preserve_lightest and len(selected_records) < color_count:
        lightest_record = max(cluster_records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(lightest_record, min_color_distance)

    while len(selected_records) < color_count:
        best_candidate = None
        best_candidate_score = None

        for record in cluster_records:
            if record["hex"] in selected_hexes:
                continue

            # Stronger distinctness rule than the default mode to avoid
            # near-duplicate clutter while keeping the palette clear.
            if not can_add_color(record["rgb"], selected_rgbs, min_color_distance):
                continue

            candidate_score = float(record["priority_score"])

            # Strongly reward dominant clusters.
            candidate_score += record["size_ratio"] * 0.30

            # Mildly reward structural lightness spread without forcing
            # arbitrary hue diversity.
            if len(selected_records) > 0:
                nearest_selected_lightness_gap = min(
                    abs(record["lightness"] - selected["lightness"])
                    for selected in selected_records
                )
                candidate_score += min(nearest_selected_lightness_gap / 255.0, 0.08)

            # Penalize tiny clusters so outlier colors do not steal slots.
            if record["size_ratio"] < 0.008:
                candidate_score -= 0.12
            elif record["size_ratio"] < 0.015:
                candidate_score -= 0.05

            # Very mild reward for moderate saturation so the palette can
            # still keep useful shading families when they are real.
            if 20.0 <= record["saturation"] <= 110.0:
                candidate_score += 0.02

            if best_candidate is None or candidate_score > best_candidate_score:
                best_candidate = record
                best_candidate_score = candidate_score

        if best_candidate is not None:
            add_record(best_candidate)
            continue

        # Relax distance only slightly. This mode should still prefer
        # clarity over stuffing in more edge-case colors.
        relaxed_threshold = max(min_color_distance * 0.65, 14.0)

        for record in cluster_records:
            if record["hex"] in selected_hexes:
                continue

            if can_add_color(record["rgb"], selected_rgbs, relaxed_threshold):
                add_record(record)
                break
        else:
            for record in cluster_records:
                if record["hex"] in selected_hexes:
                    continue

                add_record(record)
                break
            else:
                break

    return [record["hex"] for record in selected_records[:color_count]]


def select_top_frequency_low_variety_colors(
    color_stats_df,
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 26.0,
) -> list[str]:
    """
    Select a low-variety palette directly from exact source colors ranked by frequency.

    This mode is intended for simpler images where exact source colors matter
    more than cluster-based family representation.

    Strategy:
    - start from exact colors already present in the image
    - optionally keep darkest/lightest anchors
    - walk through colors in descending frequency
    - skip near-duplicates
    - lightly preserve lightness spread without forcing extra hue diversity
    """
    if color_count < 1:
        raise ValueError("color_count must be at least 1.")

    if len(color_stats_df) == 0:
        raise ValueError("No source colors were available for palette selection.")

    records = []
    total_frequency = int(color_stats_df["Frequency"].sum())

    for _, row in color_stats_df.iterrows():
        rgb = np.asarray(
            [int(row["R"]), int(row["G"]), int(row["B"])],
            dtype=np.float64,
        )

        records.append(
            {
                "rgb": rgb,
                "hex": str(row["Hex"]),
                "frequency": int(row["Frequency"]),
                "frequency_ratio": int(row["Frequency"]) / total_frequency,
                "lightness": calculate_rgb_lightness(rgb),
                "saturation": calculate_rgb_saturation(rgb),
            }
        )

    selected_records: list[dict] = []
    selected_rgbs: list[np.ndarray] = []
    selected_hexes: set[str] = set()

    def add_record(record: dict) -> None:
        selected_records.append(record)
        selected_rgbs.append(record["rgb"])
        selected_hexes.add(record["hex"])

    def try_add_anchor(record: dict, distance_threshold: float) -> bool:
        if record["hex"] in selected_hexes:
            return False

        if can_add_color(record["rgb"], selected_rgbs, distance_threshold):
            add_record(record)
            return True

        return False

    if preserve_darkest:
        darkest_record = min(records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(darkest_record, 0.0)

    if preserve_lightest and len(selected_records) < color_count:
        lightest_record = max(records, key=lambda record: (record["lightness"], record["hex"]))
        try_add_anchor(lightest_record, min_color_distance)

    while len(selected_records) < color_count:
        best_candidate = None
        best_candidate_score = None

        for record in records:
            if record["hex"] in selected_hexes:
                continue

            if not can_add_color(record["rgb"], selected_rgbs, min_color_distance):
                continue

            candidate_score = record["frequency_ratio"] * 1.0

            # Lightly reward lightness spread so simpler images can still
            # keep some structural shading without forcing unrelated colors.
            if len(selected_records) > 0:
                nearest_selected_lightness_gap = min(
                    abs(record["lightness"] - selected["lightness"])
                    for selected in selected_records
                )
                candidate_score += min(nearest_selected_lightness_gap / 255.0, 0.06)

            # Penalize tiny exact colors so stray rare pixels do not dominate.
            if record["frequency_ratio"] < 0.003:
                candidate_score -= 0.10
            elif record["frequency_ratio"] < 0.008:
                candidate_score -= 0.04

            if best_candidate is None or candidate_score > best_candidate_score:
                best_candidate = record
                best_candidate_score = candidate_score

        if best_candidate is not None:
            add_record(best_candidate)
            continue

        # Slight fallback relaxation only if strict filtering leaves us short.
        relaxed_threshold = max(min_color_distance * 0.7, 12.0)

        for record in records:
            if record["hex"] in selected_hexes:
                continue

            if can_add_color(record["rgb"], selected_rgbs, relaxed_threshold):
                add_record(record)
                break
        else:
            for record in records:
                if record["hex"] in selected_hexes:
                    continue

                add_record(record)
                break
            else:
                break

    return [record["hex"] for record in selected_records[:color_count]]


# ============================================================
# Public extraction methods
# ============================================================

def extract_top_image_palette_colors(
    image_path: Path,
    color_count: int,
) -> list[str]:
    """
    Extract the most frequent exact colors directly from the source image.

    This is the raw baseline method.
    """
    if color_count < 1:
        raise ValueError("Image-derived palette color count must be at least 1.")

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    image_array = load_image(image_path)
    color_frequencies = extract_color_frequencies(image_array)
    color_stats_df = build_color_stats_dataframe(color_frequencies)

    if color_count > len(color_stats_df):
        color_count = len(color_stats_df)

    return color_stats_df.head(color_count)["Hex"].tolist()


def extract_top_frequency_low_variety_palette_colors(
    image_path: Path,
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 26.0,
) -> list[str]:
    """
    Extract a palette for simpler images by starting from exact source-color
    frequency instead of cluster grouping.
    """
    if color_count < 1:
        raise ValueError("Image-derived palette color count must be at least 1.")

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    image_array = load_image(image_path)
    color_frequencies = extract_color_frequencies(image_array)
    color_stats_df = build_color_stats_dataframe(color_frequencies)

    if color_count > len(color_stats_df):
        color_count = len(color_stats_df)

    return select_top_frequency_low_variety_colors(
        color_stats_df=color_stats_df,
        color_count=color_count,
        preserve_darkest=preserve_darkest,
        preserve_lightest=preserve_lightest,
        min_color_distance=min_color_distance,
    )


def extract_clustered_main_palette_colors(
    image_path: Path,
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 28.0,
    random_seed: int = 42,
    representative_mode: str = "nearest_real",
    selection_mode: str = "standard",
) -> list[str]:
    """
    Extract a palette of representative "main colors" from the image.

    representative_mode:
    - nearest_real: choose the actual pixel nearest to each cluster centroid
    - most_frequent_real: choose the most common exact color inside each cluster

    selection_mode:
    - standard: prioritize strongest clusters with distance filtering
    - balanced: mildly encourage hue-family spread
    - diverse: strongly encourage hue-family spread
    - low_variety: prioritize dominant exact colors and clarity
    """
    if color_count < 1:
        raise ValueError("Image-derived palette color count must be at least 1.")

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    if selection_mode not in {"standard", "balanced", "diverse", "low_variety"}:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    image_array = load_image(image_path)
    sampled_pixels = sample_image_pixels(
        image_array=image_array,
        max_samples=50000,
        random_seed=random_seed,
    )

    internal_cluster_count = max(color_count * 3, 16)
    internal_cluster_count = min(internal_cluster_count, 100, len(sampled_pixels))

    centroids, labels = cluster_pixels_kmeans(
        pixels=sampled_pixels,
        cluster_count=internal_cluster_count,
        random_seed=random_seed,
    )

    cluster_records = build_cluster_records(
        centroids=centroids,
        labels=labels,
        sampled_pixels=sampled_pixels,
        representative_mode=representative_mode,
    )

    scored_records = score_cluster_records(cluster_records)

    if selection_mode == "balanced":
        return select_clustered_palette_colors_balanced(
            cluster_records=scored_records,
            color_count=color_count,
            preserve_darkest=preserve_darkest,
            preserve_lightest=preserve_lightest,
            min_color_distance=min_color_distance,
        )

    if selection_mode == "diverse":
        return select_clustered_palette_colors_diverse(
            cluster_records=scored_records,
            color_count=color_count,
            preserve_darkest=preserve_darkest,
            preserve_lightest=preserve_lightest,
            min_color_distance=min_color_distance,
        )

    if selection_mode == "low_variety":
        return select_clustered_palette_colors_low_variety(
            cluster_records=scored_records,
            color_count=color_count,
            preserve_darkest=preserve_darkest,
            preserve_lightest=preserve_lightest,
            min_color_distance=max(min_color_distance, 32.0),
        )

    return select_clustered_palette_colors(
        cluster_records=scored_records,
        color_count=color_count,
        preserve_darkest=preserve_darkest,
        preserve_lightest=preserve_lightest,
        min_color_distance=min_color_distance,
    )