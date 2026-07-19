import csv

from PIL import Image

from mtsd_paths_config import IMAGE_DIR, IMAGE_SIZES_CSV, make_output_dirs


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    make_output_dirs()

    image_count = 0
    with IMAGE_SIZES_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "width", "height"])

        for image_path in sorted(IMAGE_DIR.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            with Image.open(image_path) as img:
                width, height = img.size

            writer.writerow([image_path.name.lower(), width, height])
            image_count += 1

    print(f"Saved image sizes: {IMAGE_SIZES_CSV}")
    print(f"Image count: {image_count}")


if __name__ == "__main__":
    main()
