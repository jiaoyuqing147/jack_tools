from pathlib import Path

import pandas as pd

from mtsd_paths_config import (
    GT_DETECTION,
    GT_RECOGNITION,
    IMAGE_SIZES_CSV,
    LABELS_YOLO66_DIR,
    MISMATCH_CSV,
    MISSING_IMAGE_SIZE_TXT,
    MISSING_RECOGNITION_TXT,
    make_output_dirs,
)


def normalize_name(value: object) -> str:
    return str(value).strip().strip("'").strip('"').lower()


def recognition_to_scene_name(filename: str) -> str:
    # 保持原脚本逻辑：xxx_*.jpg 归到 xxx.jpg。
    stem = Path(filename).stem
    scene_stem = stem.split("_")[0]
    return f"{scene_stem}.jpg"


def main() -> None:
    make_output_dirs()

    detection_df = pd.read_csv(GT_DETECTION, sep=";", engine="python")
    recognition_df = pd.read_csv(GT_RECOGNITION, sep=";", engine="python")
    size_df = pd.read_csv(IMAGE_SIZES_CSV)

    detection_df["File Name"] = detection_df["File Name"].map(normalize_name)
    recognition_df["File Name"] = recognition_df["File Name"].map(normalize_name)
    size_df["filename"] = size_df["filename"].map(normalize_name)

    recognition_df["scene"] = recognition_df["File Name"].map(recognition_to_scene_name)
    recognition_grouped = recognition_df.groupby("scene", sort=True)

    size_map = dict(zip(size_df["filename"], zip(size_df["width"], size_df["height"])))

    missing_recog = []
    missing_size = []
    mismatches = []
    written_files = 0
    written_boxes = 0

    for scene, group in detection_df.groupby("File Name", sort=True):
        if scene not in recognition_grouped.groups:
            missing_recog.append(scene)
            continue

        if scene not in size_map:
            missing_size.append(scene)
            continue

        recog_group = recognition_grouped.get_group(scene).reset_index(drop=True)

        if len(group) != len(recog_group):
            mismatches.append((scene, len(group), len(recog_group)))
            continue

        img_w, img_h = size_map[scene]
        yolo_lines = []

        for i, det_row in enumerate(group.itertuples(index=False)):
            recog_row = recog_group.iloc[i]

            class_id = int(recog_row["Class ID"]) - 1
            x_center = (float(det_row.X) + float(det_row.Width) / 2) / img_w
            y_center = (float(det_row.Y) + float(det_row.Height) / 2) / img_h
            width_norm = float(det_row.Width) / img_w
            height_norm = float(det_row.Height) / img_h

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            )

        label_path = LABELS_YOLO66_DIR / f"{Path(scene).stem}.txt"
        label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        written_files += 1
        written_boxes += len(yolo_lines)

    MISSING_RECOGNITION_TXT.write_text("\n".join(missing_recog), encoding="utf-8")
    MISSING_IMAGE_SIZE_TXT.write_text("\n".join(missing_size), encoding="utf-8")
    pd.DataFrame(mismatches, columns=["scene", "detection_count", "recognition_count"]).to_csv(
        MISMATCH_CSV, index=False
    )

    print(f"YOLO66 labels dir: {LABELS_YOLO66_DIR}")
    print(f"Written files: {written_files}")
    print(f"Written boxes: {written_boxes}")
    print(f"Missing recognition images: {len(missing_recog)}")
    print(f"Missing image size images: {len(missing_size)}")
    print(f"Mismatch images: {len(mismatches)}")


if __name__ == "__main__":
    main()
