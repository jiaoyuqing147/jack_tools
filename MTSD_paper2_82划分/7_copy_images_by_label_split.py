import shutil
from pathlib import Path

import pandas as pd

from mtsd_case_utils import IMAGE_SUFFIXES, build_unique_stem_index, recreate_owned_directory, validate_yolo_file
from mtsd_paths_config import (
    IMAGE_DIR, REPORTS_DIR, SPLIT_IMAGES_TRAIN_DIR, SPLIT_IMAGES_VAL_DIR,
    SPLIT_LABELS_TRAIN_DIR, SPLIT_LABELS_VAL_DIR, make_output_dirs,
)


MISSING_IMAGES_CSV = REPORTS_DIR / "copy_images_missing.csv"
COPY_IMAGES_SUMMARY_TXT = REPORTS_DIR / "copy_images_summary.txt"
K = 54


def preflight(label_dir: Path, images: dict[str, Path], split: str) -> list[tuple[Path, Path]]:
    labels = build_unique_stem_index(label_dir, {".txt"})
    pairs = []
    missing = []
    for key, label in labels.items():
        validate_yolo_file(label, K, allow_empty=False)
        image = images.get(key)
        if image is None:
            missing.append({"split": split, "label_file": label.name, "image_stem": label.stem})
            continue
        if label.stem != image.stem:
            raise RuntimeError(f"Non-exact image/label stem before copy: {image.name} / {label.name}")
        pairs.append((label, image))
    if missing:
        pd.DataFrame(missing).to_csv(MISSING_IMAGES_CSV, index=False)
        raise FileNotFoundError(f"{len(missing)} labels have no source image; see {MISSING_IMAGES_CSV}")
    return pairs


def copy_and_verify(pairs: list[tuple[Path, Path]], destination: Path) -> None:
    recreate_owned_directory(destination)
    for _, image in pairs:
        shutil.copy2(image, destination / image.name)
    outputs = build_unique_stem_index(destination, IMAGE_SUFFIXES)
    expected = {image.stem.casefold(): image.name for _, image in pairs}
    actual = {key: path.name for key, path in outputs.items()}
    if actual != expected:
        raise RuntimeError(f"Copied image output mismatch in {destination}")


def main() -> None:
    make_output_dirs()
    images = build_unique_stem_index(IMAGE_DIR, IMAGE_SUFFIXES)
    train = preflight(SPLIT_LABELS_TRAIN_DIR, images, "train")
    val = preflight(SPLIT_LABELS_VAL_DIR, images, "val")
    train_keys = {image.stem.casefold() for _, image in train}
    val_keys = {image.stem.casefold() for _, image in val}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Train/val overlap ignoring case: {sorted(overlap)[:10]}")
    copy_and_verify(train, SPLIT_IMAGES_TRAIN_DIR)
    copy_and_verify(val, SPLIT_IMAGES_VAL_DIR)
    if MISSING_IMAGES_CSV.exists():
        MISSING_IMAGES_CSV.unlink()
    summary = [
        "MTSD YOLO54 paper2 copy images summary", f"source_images: {IMAGE_DIR}",
        f"train_images_copied: {len(train)}", f"val_images_copied: {len(val)}",
        f"total_images_copied: {len(train) + len(val)}", "missing_images: 0",
    ]
    COPY_IMAGES_SUMMARY_TXT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
