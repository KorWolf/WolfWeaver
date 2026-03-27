from pathlib import Path
from PIL import Image
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image and return as a NumPy array (H, W, 3)
    """
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def get_image_dimensions(image_array: np.ndarray) -> tuple[int, int]:
    """
    Return (height, width)
    """
    height, width, _ = image_array.shape
    return height, width

