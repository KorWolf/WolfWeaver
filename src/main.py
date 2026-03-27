from pathlib import Path

from src.image_io import load_image, get_image_dimensions
from src.color_stats import extract_color_frequencies, rgb_to_hex


def main() -> None:
    print("Tapestry pipeline starting...")

    image_path = Path("input/source_images/birthOfVenus.png")

    # Load image
    image_array = load_image(image_path)

    # Dimensions
    height, width = get_image_dimensions(image_array)
    print(f"Image size: {width} x {height}")

    # Extract color frequencies
    color_freq = extract_color_frequencies(image_array)

    print(f"Unique colors: {len(color_freq)}")

    # Print sample
    sample_items = list(color_freq.items())[:10]

    print("\nSample colors:")
    for rgb, count in sample_items:
        print(f"{rgb_to_hex(rgb)} -> {count}")


if __name__ == "__main__":
    main()

