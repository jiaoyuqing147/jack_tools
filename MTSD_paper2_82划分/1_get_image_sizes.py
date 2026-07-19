import csv

from PIL import Image

from mtsd_case_utils import IMAGE_SUFFIXES, build_unique_stem_index
from mtsd_paths_config import IMAGE_DIR, IMAGE_SIZES_CSV, make_output_dirs


def main() -> None:
    make_output_dirs()
    image_index = build_unique_stem_index(IMAGE_DIR, IMAGE_SUFFIXES)
    rows = []
    for image_path in sorted(image_index.values(), key=lambda p: (p.name.casefold(), p.name)):
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {image_path}: {width}x{height}")
        rows.append((image_path.name, width, height))  # real on-disk name, never lookup key

    IMAGE_SIZES_CSV.parent.mkdir(parents=True, exist_ok=True)
    temp_path = IMAGE_SIZES_CSV.with_suffix(IMAGE_SIZES_CSV.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "width", "height"])
        writer.writerows(rows)
    temp_path.replace(IMAGE_SIZES_CSV)

    with IMAGE_SIZES_CSV.open(newline="", encoding="utf-8") as csvfile:
        written = list(csv.DictReader(csvfile))
    if [row["filename"] for row in written] != [row[0] for row in rows]:
        raise RuntimeError("image_sizes.csv verification failed: filenames changed while writing")
    print(f"Saved image sizes: {IMAGE_SIZES_CSV}")
    print(f"Image count: {len(rows)}")


if __name__ == "__main__":
    main()
