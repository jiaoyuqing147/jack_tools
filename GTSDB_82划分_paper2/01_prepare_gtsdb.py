import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from config import (
    CONVERT_TO_PNG,
    GT_TXT_PATH,
    NUM_CLASSES,
    PNG_COMPRESSION,
    SOURCE_IMAGE_DIR,
    WORK_DIR,
)

SUPPORTED_EXTENSIONS = {".ppm", ".png", ".jpg", ".jpeg", ".bmp"}
IMAGE_DIR = WORK_DIR / "images_all"
LABEL_DIR = WORK_DIR / "labels_all"
REPORT_DIR = WORK_DIR / "reports"


def scan_images(directory: Path):
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    by_stem = {}
    for path in files:
        if path.stem in by_stem:
            raise RuntimeError(
                f"发现同名但扩展名不同的图片：{by_stem[path.stem]} 和 {path}"
            )
        by_stem[path.stem] = path
    return by_stem


def read_annotations(gt_path: Path):
    annotations = defaultdict(list)
    malformed = []
    with gt_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) < 6:
                malformed.append(f"第{line_no}行字段不足：{row}")
                continue
            try:
                name = Path(row[0].strip()).stem
                x1, y1, x2, y2 = map(int, row[1:5])
                class_id = int(row[5])
            except ValueError:
                malformed.append(f"第{line_no}行无法解析：{row}")
                continue
            annotations[name].append((line_no, x1, y1, x2, y2, class_id))
    return annotations, malformed


def reset_prepared_dirs():
    # 这些是本流程专用的中间目录，每次运行都重新生成。
    for directory in (IMAGE_DIR, LABEL_DIR, REPORT_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def main():
    if not SOURCE_IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"原始图片目录不存在：{SOURCE_IMAGE_DIR}")
    if not GT_TXT_PATH.is_file():
        raise FileNotFoundError(f"gt.txt不存在：{GT_TXT_PATH}")

    image_map = scan_images(SOURCE_IMAGE_DIR)
    annotations, errors = read_annotations(GT_TXT_PATH)
    if not image_map:
        raise RuntimeError("没有找到任何支持的图片。")

    reset_prepared_dirs()
    class_box_counts = [0] * NUM_CLASSES
    positive_images = 0

    unknown_annotation_images = sorted(set(annotations) - set(image_map))
    for stem in unknown_annotation_images:
        errors.append(f"标注存在但图片不存在：{stem}")

    for index, (stem, source_path) in enumerate(image_map.items(), start=1):
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f"图片读取失败：{source_path}")
            continue
        height, width = image.shape[:2]

        if CONVERT_TO_PNG:
            output_image = IMAGE_DIR / f"{stem}.png"
            ok = cv2.imwrite(
                str(output_image), image,
                [int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION],
            )
            if not ok:
                errors.append(f"PNG保存失败：{output_image}")
                continue
        else:
            output_image = IMAGE_DIR / source_path.name
            shutil.copy2(source_path, output_image)

        yolo_lines = []
        valid_boxes = 0
        seen_boxes = set()
        for line_no, x1, y1, x2, y2, class_id in annotations.get(stem, []):
            box_key = (x1, y1, x2, y2, class_id)
            if box_key in seen_boxes:
                errors.append(f"第{line_no}行重复标注：{stem} {box_key}")
                continue
            seen_boxes.add(box_key)

            if not 0 <= class_id < NUM_CLASSES:
                errors.append(f"第{line_no}行类别越界：class_id={class_id}")
                continue
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append(
                    f"第{line_no}行边界框非法：{stem} "
                    f"({x1},{y1},{x2},{y2}), image={width}x{height}"
                )
                continue

            x_center = ((x1 + x2) / 2.0) / width
            y_center = ((y1 + y2) / 2.0) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f}\n"
            )
            class_box_counts[class_id] += 1
            valid_boxes += 1

        if valid_boxes:
            positive_images += 1
        (LABEL_DIR / f"{stem}.txt").write_text("".join(yolo_lines), encoding="utf-8")

        if index % 100 == 0 or index == len(image_map):
            print(f"处理进度：{index}/{len(image_map)}")

    prepared_images = list(IMAGE_DIR.iterdir())
    summary = [
        f"原始图片数量：{len(image_map)}",
        f"成功处理图片数量：{len(prepared_images)}",
        f"有目标图片数量：{positive_images}",
        f"背景图片数量：{len(prepared_images) - positive_images}",
        f"合法目标框数量：{sum(class_box_counts)}",
        f"出现的类别数量：{sum(count > 0 for count in class_box_counts)}/{NUM_CLASSES}",
        f"错误或警告数量：{len(errors)}",
    ]
    (REPORT_DIR / "prepare_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    (REPORT_DIR / "prepare_errors.txt").write_text(
        "\n".join(errors) + ("\n" if errors else ""), encoding="utf-8"
    )
    with (REPORT_DIR / "class_box_counts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "box_count"])
        writer.writerows(enumerate(class_box_counts))

    print("\n".join(summary))
    print(f"处理结果：{WORK_DIR}")
    if errors:
        print(f"请检查：{REPORT_DIR / 'prepare_errors.txt'}")


if __name__ == "__main__":
    main()

