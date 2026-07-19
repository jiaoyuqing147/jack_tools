from pathlib import Path

import pandas as pd

from mtsd_case_utils import (
    IMAGE_SUFFIXES,
    assert_exact_output,
    build_unique_stem_index,
    case_key,
    clear_owned_files,
    validate_yolo_file,
)
from mtsd_paths_config import (
    GT_DETECTION, GT_RECOGNITION, IMAGE_DIR, IMAGE_SIZES_CSV, LABELS_YOLO66_DIR,
    MISMATCH_CSV, MISSING_IMAGE_SIZE_TXT, MISSING_RECOGNITION_TXT, make_output_dirs,
)


def clean_name(value: object) -> str:
    return str(value).strip().strip("'").strip('"')


def recognition_scene_key(value: object) -> str:
    stem = Path(clean_name(value)).stem
    return f"{stem.split('_')[0]}.jpg".casefold()


def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} missing columns: {sorted(missing)}")


def main() -> None:
    make_output_dirs()
    for source in (GT_DETECTION, GT_RECOGNITION, IMAGE_SIZES_CSV):
        if not source.is_file():
            raise FileNotFoundError(source)

    images = build_unique_stem_index(IMAGE_DIR, IMAGE_SUFFIXES)
    detection = pd.read_csv(GT_DETECTION, sep=";", engine="python")
    recognition = pd.read_csv(GT_RECOGNITION, sep=";", engine="python")
    sizes = pd.read_csv(IMAGE_SIZES_CSV)
    require_columns(detection, {"File Name", "X", "Y", "Width", "Height"}, GT_DETECTION)
    require_columns(recognition, {"File Name", "Class ID"}, GT_RECOGNITION)
    require_columns(sizes, {"filename", "width", "height"}, IMAGE_SIZES_CSV)

    size_map: dict[str, tuple[Path, float, float]] = {}
    for row in sizes.itertuples(index=False):
        csv_name = clean_name(row.filename)
        key = Path(csv_name).stem.casefold()
        image_path = images.get(key)
        if image_path is None:
            raise FileNotFoundError(f"image_sizes.csv references missing source image: {csv_name}")
        if csv_name != image_path.name:
            raise RuntimeError(f"image_sizes.csv does not preserve real case: {csv_name} != {image_path.name}")
        if key in size_map:
            raise RuntimeError(f"Duplicate image_sizes.csv stem ignoring case: {csv_name}")
        size_map[key] = (image_path, float(row.width), float(row.height))
    if set(size_map) != set(images):
        missing = sorted(images[key].name for key in set(images) - set(size_map))
        raise RuntimeError(f"image_sizes.csv is incomplete; missing {len(missing)} images: {missing[:10]}")

    detection["scene_key"] = detection["File Name"].map(lambda v: Path(clean_name(v)).stem.casefold())
    recognition["scene_key"] = recognition["File Name"].map(recognition_scene_key).map(lambda v: Path(v).stem.casefold())
    recognition_groups = recognition.groupby("scene_key", sort=True)

    # Clear only after every source/index preflight succeeds.
    clear_owned_files(LABELS_YOLO66_DIR, {".txt"})
    missing_recognition: list[str] = []
    missing_size: list[str] = []
    mismatches: list[tuple[str, int, int]] = []
    expected_names: list[str] = []
    box_count = 0

    for key, det_group in detection.groupby("scene_key", sort=True):
        if key not in recognition_groups.groups:
            missing_recognition.append(key)
            continue
        if key not in size_map:
            missing_size.append(key)
            continue
        rec_group = recognition_groups.get_group(key).reset_index(drop=True)
        det_group = det_group.reset_index(drop=True)
        if len(det_group) != len(rec_group):
            mismatches.append((key, len(det_group), len(rec_group)))
            continue
        real_image, image_width, image_height = size_map[key]
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid size for {real_image}: {image_width}x{image_height}")
        lines = []
        for index, det in enumerate(det_group.itertuples(index=False)):
            class_id = int(rec_group.iloc[index]["Class ID"]) - 1
            x, y = float(det.X), float(det.Y)
            width, height = float(det.Width), float(det.Height)
            if not 0 <= class_id < 66 or width <= 0 or height <= 0:
                raise ValueError(f"Invalid annotation for {real_image}, row {index}")
            coords = ((x + width / 2) / image_width, (y + height / 2) / image_height,
                      width / image_width, height / image_height)
            if not all(0.0 <= value <= 1.0 for value in coords):
                raise ValueError(f"Normalized coordinate outside [0,1] for {real_image}: {coords}")
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in coords))
        output_name = f"{real_image.stem}.txt"  # authoritative real-image stem
        output_path = LABELS_YOLO66_DIR / output_name
        if output_path.exists():
            raise RuntimeError(f"Duplicate output label: {output_path}")
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        validate_yolo_file(output_path, 66)
        expected_names.append(output_name)
        box_count += len(lines)

    MISSING_RECOGNITION_TXT.write_text("\n".join(missing_recognition) + ("\n" if missing_recognition else ""), encoding="utf-8")
    MISSING_IMAGE_SIZE_TXT.write_text("\n".join(missing_size) + ("\n" if missing_size else ""), encoding="utf-8")
    pd.DataFrame(mismatches, columns=["scene", "detection_count", "recognition_count"]).to_csv(MISMATCH_CSV, index=False)
    assert_exact_output(LABELS_YOLO66_DIR, expected_names)
    print(f"YOLO66 labels: {len(expected_names)}, boxes: {box_count}")
    print(f"Missing recognition: {len(missing_recognition)}, missing size: {len(missing_size)}, mismatches: {len(mismatches)}")


if __name__ == "__main__":
    main()
