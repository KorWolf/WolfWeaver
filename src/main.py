from pathlib import Path


def main() -> None:
    print("Tapestry pipeline starting...")

    required_dirs = [
        Path("input/source_images"),
        Path("input/palettes"),
        Path("output/frames"),
        Path("output/gifs"),
        Path("output/tables"),
        Path("output/debug"),
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    print("Project folders verified.")
    print("Setup looks good.")


if __name__ == "__main__":
    main()

