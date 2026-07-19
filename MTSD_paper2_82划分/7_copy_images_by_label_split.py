import shutil
from pathlib import Path

import pandas as pd

from mtsd_paths_config import (
    IMAGE_DIR,
    REPORTS_DIR,
    SPLIT_IMAGES_TRAIN_DIR,
    SPLIT_IMAGES_VAL_DIR,
    SPLIT_LABELS_TRAIN_DIR,
    SPLIT_LABELS_VAL_DIR,
    make_output_dirs,
)


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MISSING_IMAGES_CSV = REPORTS_DIR / "copy_images_missing.csv"
COPY_IMAGES_SUMMARY_TXT = REPORTS_DIR / "copy_images_summary.txt"


def build_image_index(image_dir: Path) -> dict[str, Path]:
    image_index = {}
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            image_index[image_path.stem.lower()] = image_path
    return image_index


def clear_image_dir(image_dir: Path) -> None:
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)


def copy_images_for_split(label_dir: Path, image_dst_dir: Path, image_index: dict[str, Path], split: str):
    copied_rows = []
    missing_rows = []

    for label_path in sorted(label_dir.glob("*.txt")):
        stem = label_path.stem.lower()
        src_image = image_index.get(stem)

        if src_image is None:
            missing_rows.append({
                "split": split,
                "label_file": label_path.name,
                "image_stem": label_path.stem,
            })
            continue

        dst_image = image_dst_dir / src_image.name
        shutil.copy2(src_image, dst_image)
        copied_rows.append({
            "split": split,
            "label_file": label_path.name,
            "image_file": src_image.name,
            "src_image": str(src_image),
            "dst_image": str(dst_image),
        })

    return copied_rows, missing_rows


def main() -> None:
    make_output_dirs()
    clear_image_dir(SPLIT_IMAGES_TRAIN_DIR)
    clear_image_dir(SPLIT_IMAGES_VAL_DIR)

    image_index = build_image_index(IMAGE_DIR)

    train_copied, train_missing = copy_images_for_split(
        SPLIT_LABELS_TRAIN_DIR, SPLIT_IMAGES_TRAIN_DIR, image_index, "train"
    )
    val_copied, val_missing = copy_images_for_split(
        SPLIT_LABELS_VAL_DIR, SPLIT_IMAGES_VAL_DIR, image_index, "val"
    )

    copied_rows = train_copied + val_copied
    missing_rows = train_missing + val_missing

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(MISSING_IMAGES_CSV, index=False)
        raise FileNotFoundError(
            f"有 {len(missing_rows)} 个标签找不到对应图片，详情见: {MISSING_IMAGES_CSV}"
        )

    summary_lines = [
        "MTSD YOLO54 paper2 copy images summary",
        f"source_images: {IMAGE_DIR}",
        f"train_label_dir: {SPLIT_LABELS_TRAIN_DIR}",
        f"val_label_dir: {SPLIT_LABELS_VAL_DIR}",
        f"train_image_dir: {SPLIT_IMAGES_TRAIN_DIR}",
        f"val_image_dir: {SPLIT_IMAGES_VAL_DIR}",
        f"train_images_copied: {len(train_copied)}",
        f"val_images_copied: {len(val_copied)}",
        f"total_images_copied: {len(copied_rows)}",
        f"missing_images: {len(missing_rows)}",
    ]
    COPY_IMAGES_SUMMARY_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
