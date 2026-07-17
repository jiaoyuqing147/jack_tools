import csv
import shutil
from pathlib import Path

import cv2

from config import CLEAN_OUTPUT, NUM_CLASSES, OUTPUT_DATASET_DIR, WORK_DIR

SOURCE_IMAGE_DIR = WORK_DIR / "images_all"
SOURCE_LABEL_DIR = WORK_DIR / "labels_all"
SPLIT_DIR = WORK_DIR / "splits"


def read_split(name):
    path = SPLIT_DIR / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"划分列表不存在：{path}")
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"{path} 内部存在重复图片名。")
    return stems


def find_image(stem):
    matches = list(SOURCE_IMAGE_DIR.glob(f"{stem}.*"))
    if len(matches) != 1:
        raise RuntimeError(f"图片 {stem} 匹配数量不是1：{matches}")
    return matches[0]


def prepare_output():
    managed = [
        OUTPUT_DATASET_DIR / "images",
        OUTPUT_DATASET_DIR / "labels",
        OUTPUT_DATASET_DIR / "splits",
        OUTPUT_DATASET_DIR / "reports",
    ]
    nonempty = [p for p in managed if p.exists() and any(p.iterdir())]
    if nonempty and not CLEAN_OUTPUT:
        raise RuntimeError(
            "输出目录已有文件。为避免混入旧划分，请更换 OUTPUT_DATASET_DIR；"
            "确认只清理本流程目录时，也可把 CLEAN_OUTPUT 改为 True。\n"
            + "\n".join(map(str, nonempty))
        )
    if CLEAN_OUTPUT:
        for path in managed:
            if path.exists():
                shutil.rmtree(path)
    for split in ("train", "val"):
        (OUTPUT_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET_DIR / "splits").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET_DIR / "reports").mkdir(parents=True, exist_ok=True)


def copy_split(name, stems):
    for index, stem in enumerate(stems, start=1):
        image = find_image(stem)
        label = SOURCE_LABEL_DIR / f"{stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"标签不存在：{label}")
        shutil.copy2(image, OUTPUT_DATASET_DIR / "images" / name / image.name)
        shutil.copy2(label, OUTPUT_DATASET_DIR / "labels" / name / label.name)
        if index % 100 == 0 or index == len(stems):
            print(f"复制{name}：{index}/{len(stems)}")


def verify_split(name, expected_stems):
    image_dir = OUTPUT_DATASET_DIR / "images" / name
    label_dir = OUTPUT_DATASET_DIR / "labels" / name
    images = {p.stem: p for p in image_dir.iterdir() if p.is_file()}
    labels = {p.stem: p for p in label_dir.glob("*.txt")}
    errors = []
    if set(images) != set(expected_stems):
        errors.append(f"{name}图片集合与划分列表不一致")
    if set(labels) != set(expected_stems):
        errors.append(f"{name}标签集合与划分列表不一致")

    class_images = [0] * NUM_CLASSES
    class_boxes = [0] * NUM_CLASSES
    background = 0
    for stem, image_path in images.items():
        if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
            errors.append(f"损坏图片：{image_path}")
        label_path = labels.get(stem)
        if label_path is None:
            continue
        present = set()
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"标签格式错误：{label_path} 第{line_no}行")
                continue
            try:
                class_id = int(parts[0])
                coordinates = list(map(float, parts[1:]))
            except ValueError:
                errors.append(f"标签无法解析：{label_path} 第{line_no}行")
                continue
            if not 0 <= class_id < NUM_CLASSES or not all(0.0 <= v <= 1.0 for v in coordinates):
                errors.append(f"标签数值越界：{label_path} 第{line_no}行")
                continue
            present.add(class_id)
            class_boxes[class_id] += 1
        if not present:
            background += 1
        for class_id in present:
            class_images[class_id] += 1
    return errors, class_images, class_boxes, background


def main():
    train = read_split("train")
    val = read_split("val")
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise RuntimeError(f"train和val存在重复：{overlap[:20]}")

    prepare_output()
    copy_split("train", train)
    copy_split("val", val)

    all_errors = []
    stats = {}
    for name, stems in (("train", train), ("val", val)):
        errors, image_counts, box_counts, background = verify_split(name, stems)
        all_errors.extend(errors)
        stats[name] = (image_counts, box_counts, background)

    existing_classes = {
        c for c in range(NUM_CLASSES)
        if stats["train"][0][c] + stats["val"][0][c] > 0
    }
    train_classes = {c for c in existing_classes if stats["train"][0][c] > 0}
    val_classes = {c for c in existing_classes if stats["val"][0][c] > 0}
    splittable_classes = {
        c for c in existing_classes
        if stats["train"][0][c] + stats["val"][0][c] >= 2
    }
    train_missing = sorted(existing_classes - train_classes)
    val_missing = sorted(existing_classes - val_classes)
    avoidable_val_missing = sorted(splittable_classes - val_classes)
    if train_missing:
        all_errors.append(f"train缺失已有类别：{train_missing}")
    if avoidable_val_missing:
        all_errors.append(
            f"这些类别至少有2张图片，但val仍然缺失：{avoidable_val_missing}"
        )

    report_path = OUTPUT_DATASET_DIR / "reports" / "final_class_statistics.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "train_images", "val_images", "train_boxes", "val_boxes"
        ])
        for class_id in range(NUM_CLASSES):
            writer.writerow([
                class_id,
                stats["train"][0][class_id], stats["val"][0][class_id],
                stats["train"][1][class_id], stats["val"][1][class_id],
            ])

    shutil.copy2(SPLIT_DIR / "train.txt", OUTPUT_DATASET_DIR / "splits" / "train.txt")
    shutil.copy2(SPLIT_DIR / "val.txt", OUTPUT_DATASET_DIR / "splits" / "val.txt")
    source_stats = WORK_DIR / "reports" / "split_class_statistics.csv"
    if source_stats.exists():
        shutil.copy2(source_stats, OUTPUT_DATASET_DIR / "reports" / source_stats.name)

    yaml_text = (
        f"path: {OUTPUT_DATASET_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {NUM_CLASSES}\n"
        f"names: {[str(i) for i in range(NUM_CLASSES)]}\n"
    )
    (OUTPUT_DATASET_DIR / "gtsdb.yaml").write_text(yaml_text, encoding="utf-8")

    summary = [
        f"train图片/标签：{len(train)}/{len(train)}",
        f"val图片/标签：{len(val)}/{len(val)}",
        f"train背景图片：{stats['train'][2]}",
        f"val背景图片：{stats['val'][2]}",
        f"数据中实际类别：{len(existing_classes)}/{NUM_CLASSES}",
        f"train类别覆盖：{len(train_classes)}/{len(existing_classes)}",
        f"val类别覆盖：{len(val_classes)}/{len(existing_classes)}",
        f"val缺失类别：{val_missing}",
        f"可划分但val仍缺失类别：{avoidable_val_missing}",
        "train/val重复：0",
        f"验收错误：{len(all_errors)}",
    ]
    (OUTPUT_DATASET_DIR / "reports" / "final_verification.txt").write_text(
        "\n".join(summary + (["", *all_errors] if all_errors else [])) + "\n",
        encoding="utf-8",
    )
    print("\n".join(summary))
    if all_errors:
        raise RuntimeError("最终验收失败，请查看 reports/final_verification.txt")
    print(f"最终数据集：{OUTPUT_DATASET_DIR}")


if __name__ == "__main__":
    main()
