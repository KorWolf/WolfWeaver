from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.color_stats import build_color_stats_dataframe, extract_color_frequencies
from src.image_io import load_image


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


def sample_image_pixels(image_array: np.ndarray, max_samples: int = 50000, random_seed: int = 42) -> np.ndarray:
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


def build_cluster_records(
    centroids: np.ndarray,
    labels: np.ndarray,
    sampled_pixels: np.ndarray,
) -> list[dict]:
    """
    Build metadata records for each cluster.

    Each record includes:
    - centroid RGB
    - hex
    - pixel count
    - size ratio
    - saturation
    - lightness
    - distinctiveness from global mean
    """
    total_pixels = len(sampled_pixels)
    global_mean = np.mean(sampled_pixels, axis=0)

    records: list[dict] = []

    for cluster_index, centroid in enumerate(centroids):
        count = int(np.sum(labels == cluster_index))
        if count <= 0:
            continue

        rgb = np.asarray(centroid, dtype=np.float64)
        saturation = calculate_rgb_saturation(rgb)
        lightness = calculate_rgb_lightness(rgb)
        distance_from_global_mean = calculate_rgb_distance(rgb, global_mean)

        records.append(
            {
                "rgb": rgb,
                "hex": rgb_array_to_hex(rgb),
                "count": count,
                "size_ratio": count / total_pixels,
                "saturation": saturation,
                "lightness": lightness,
                "distance_from_global_mean": distance_from_global_mean,
            }
        )

    return records


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


def can_add_color(candidate_rgb: np.ndarray, selected_rgbs: list[np.ndarray], min_color_distance: float) -> bool:
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


def extract_clustered_main_palette_colors(
    image_path: Path,
    color_count: int,
    preserve_darkest: bool = True,
    preserve_lightest: bool = True,
    min_color_distance: float = 28.0,
    random_seed: int = 42,
) -> list[str]:
    """
    Extract a palette of representative "main colors" from the image.

    This method:
    - clusters similar colors together
    - ranks clusters by a balanced score
    - selects final colors with minimum-distance filtering

    This is intended to be more user-friendly than raw top-frequency picking.
    """
    if color_count < 1:
        raise ValueError("Image-derived palette color count must be at least 1.")

    if not image_path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    image_array = load_image(image_path)
    sampled_pixels = sample_image_pixels(
        image_array=image_array,
        max_samples=50000,
        random_seed=random_seed,
    )

    internal_cluster_count = max(color_count * 3, 16)
    internal_cluster_count = min(internal_cluster_count, 48, len(sampled_pixels))

    centroids, labels = cluster_pixels_kmeans(
        pixels=sampled_pixels,
        cluster_count=internal_cluster_count,
        random_seed=random_seed,
    )

    cluster_records = build_cluster_records(
        centroids=centroids,
        labels=labels,
        sampled_pixels=sampled_pixels,
    )

    scored_records = score_cluster_records(cluster_records)

    return select_clustered_palette_colors(
        cluster_records=scored_records,
        color_count=color_count,
        preserve_darkest=preserve_darkest,
        preserve_lightest=preserve_lightest,
        min_color_distance=min_color_distance,
    )