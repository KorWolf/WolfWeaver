import numpy as np
from collections import Counter


def extract_color_frequencies(image_array: np.ndarray) -> dict[tuple[int, int, int], int]:
    """
    Count how many times each RGB color appears in the image
    """
    # reshape to list of pixels
    pixels = image_array.reshape(-1, 3)

    # convert to tuples so they can be counted
    pixel_tuples = [tuple(pixel) for pixel in pixels]

    return dict(Counter(pixel_tuples))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """
    Convert (R, G, B) to hex string
    """
    return "#{:02X}{:02X}{:02X}".format(*rgb)

