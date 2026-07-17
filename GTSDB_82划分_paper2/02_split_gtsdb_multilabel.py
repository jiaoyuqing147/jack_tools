import csv
from collections import Counter
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from config import NUM_CLASSES, RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, WORK_DIR

IMAGE_DIR = WORK_DIR / "images_all"
LABEL_DIR = WORK_DIR / "labels_all"
SPLIT_DIR = WORK_DIR / "splits"
REPORT_DIR = WORK_DIR / "reports"


def read_label(label_path: Path):
    classes = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path} 第{line_no}行不是5列：{line}")
        class_id = int(parts[0])
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"{label_path} 第{line_no}行类别越界：{class_id}")
        classes.append(class_id)
    return classes


def load_samples():
    image_paths = sorted(p for p in IMAGE_DIR.iterdir() if p.is_file())
    samples = []
    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"图片没有同名标签：{image_path}")
        box_classes = read_label(label_path)
        samples.append({
            "stem": image_path.stem,
            "classes": set(box_classes),
            "box_classes": box_classes,
        })
    return samples


def multilabel_split(samples):
    total = len(samples)
    desired_val = int(round(total * VAL_RATIO))

    class_to_indices = {c: [] for c in range(NUM_CLASSES)}
    for i, sample in enumerate(samples):
        for c in sample["classes"]:
            class_to_indices[c].append(i)

    # 只出现于一张图片的类别必须留在训练集。
    forced_train = {
        indices[0]
        for indices in class_to_indices.values()
        if len(indices) == 1
    }
    candidates = [i for i in range(total) if i not in forced_train]
    if desired_val >= len(candidates):
        raise RuntimeError("可划分样本过少，无法生成目标大小的验证集。")

    # 43个类别 + 1个背景类别，保证无目标图片也按比例划分。
    y = np.zeros((len(candidates), NUM_CLASSES + 1), dtype=np.uint8)
    for row, sample_index in enumerate(candidates):
        classes = samples[sample_index]["classes"]
        if classes:
            y[row, list(classes)] = 1
        else:
            y[row, NUM_CLASSES] = 1

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        # 传入整数可避免浮点比例在小数据集上向上取整多出1张。
        test_size=desired_val,
        random_state=RANDOM_SEED,
    )
    x_dummy = np.zeros((len(candidates), 1), dtype=np.uint8)
    train_local, val_local = next(splitter.split(x_dummy, y))
    train = set(forced_train) | {candidates[i] for i in train_local}
    val = {candidates[i] for i in val_local}

    # 某些版本的底层划分器对测试集大小仍可能向上取整；优先移走背景图，
    # 或移走不会导致val丢失类别的图片，使总数量严格接近8:2。
    while len(val) > desired_val:
        val_class_image_counts = Counter(
            c for i in val for c in samples[i]["classes"]
        )
        safe = [
            i for i in val
            if all(val_class_image_counts[c] >= 2 for c in samples[i]["classes"])
        ]
        if not safe:
            break
        move_index = min(
            safe,
            key=lambda i: (
                bool(samples[i]["classes"]),
                len(samples[i]["classes"]),
                samples[i]["stem"],
            ),
        )
        val.remove(move_index)
        train.add(move_index)

    # 安全修复：如果某个已存在类别意外没有进入train，将一张对应val图片移入train。
    for class_id, indices in class_to_indices.items():
        if indices and not any(i in train for i in indices):
            move_index = min(indices, key=lambda i: len(samples[i]["classes"]))
            val.remove(move_index)
            train.add(move_index)

    # 覆盖修复：一个类别只要至少存在于2张图片，就尽量保证val至少包含1张。
    # 移动前必须确认该图片离开train后，不会让train丢失它包含的任何类别。
    for class_id, indices in class_to_indices.items():
        if len(indices) < 2 or any(i in val for i in indices):
            continue
        train_class_image_counts = Counter(
            c for i in train for c in samples[i]["classes"]
        )
        safe_candidates = [
            i for i in indices
            if i in train
            and i not in forced_train
            and all(train_class_image_counts[c] >= 2 for c in samples[i]["classes"])
        ]
        if safe_candidates:
            # 优先选择包含类别较少的图片，降低对其他类别比例的扰动。
            move_index = min(
                safe_candidates,
                key=lambda i: (len(samples[i]["classes"]), samples[i]["stem"]),
            )
            train.remove(move_index)
            val.add(move_index)

    # 覆盖修复可能让val超过目标数量。优先将背景图移回train；其次只移动
    # 不会造成val类别缺失的图片。这样既恢复8:2，又保留最大类别覆盖。
    while len(val) > desired_val:
        val_class_image_counts = Counter(
            c for i in val for c in samples[i]["classes"]
        )
        safe = [
            i for i in val
            if all(val_class_image_counts[c] >= 2 for c in samples[i]["classes"])
        ]
        if not safe:
            break
        move_index = min(
            safe,
            key=lambda i: (
                bool(samples[i]["classes"]),
                len(samples[i]["classes"]),
                samples[i]["stem"],
            ),
        )
        val.remove(move_index)
        train.add(move_index)

    # 尽量恢复验证集目标大小：只移动不会造成train类别丢失的样本。
    while len(val) < desired_val:
        train_class_image_counts = Counter(
            c for i in train for c in samples[i]["classes"]
        )
        safe = [
            i for i in train
            if i not in forced_train
            and all(train_class_image_counts[c] >= 2 for c in samples[i]["classes"])
        ]
        if not safe:
            break
        # 优先移动背景图，其次移动类别较少的图片。
        move_index = min(safe, key=lambda i: (bool(samples[i]["classes"]), len(samples[i]["classes"]), samples[i]["stem"]))
        train.remove(move_index)
        val.add(move_index)

    return sorted(train), sorted(val), class_to_indices, forced_train


def write_list(path: Path, indices, samples):
    path.write_text(
        "".join(f"{samples[i]['stem']}\n" for i in indices),
        encoding="utf-8",
    )


def make_statistics(samples, train, val):
    rows = []
    train_set, val_set = set(train), set(val)
    for class_id in range(NUM_CLASSES):
        total_images = sum(class_id in s["classes"] for s in samples)
        train_images = sum(class_id in samples[i]["classes"] for i in train_set)
        val_images = sum(class_id in samples[i]["classes"] for i in val_set)
        total_boxes = sum(s["box_classes"].count(class_id) for s in samples)
        train_boxes = sum(samples[i]["box_classes"].count(class_id) for i in train_set)
        val_boxes = sum(samples[i]["box_classes"].count(class_id) for i in val_set)
        rows.append([
            class_id, total_images, train_images, val_images,
            total_boxes, train_boxes, val_boxes,
            f"{(val_images / total_images):.4f}" if total_images else "",
        ])
    return rows


def main():
    if abs(TRAIN_RATIO + VAL_RATIO - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VAL_RATIO 必须等于1。")
    if not IMAGE_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError("请先运行 01_prepare_gtsdb.py。")

    samples = load_samples()
    train, val, class_to_indices, forced_train = multilabel_split(samples)
    train_set, val_set = set(train), set(val)
    if train_set & val_set:
        raise RuntimeError("严重错误：train和val发生重叠。")
    if train_set | val_set != set(range(len(samples))):
        raise RuntimeError("严重错误：有图片未被划分。")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_list(SPLIT_DIR / "train.txt", train, samples)
    write_list(SPLIT_DIR / "val.txt", val, samples)

    rows = make_statistics(samples, train, val)
    with (REPORT_DIR / "split_class_statistics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "total_images", "train_images", "val_images",
            "total_boxes", "train_boxes", "val_boxes", "val_image_ratio",
        ])
        writer.writerows(rows)

    train_classes = {row[0] for row in rows if row[2] > 0}
    val_classes = {row[0] for row in rows if row[3] > 0}
    existing_classes = {c for c, indices in class_to_indices.items() if indices}
    splittable_classes = {c for c, indices in class_to_indices.items() if len(indices) >= 2}
    background_total = sum(not s["classes"] for s in samples)
    background_train = sum(not samples[i]["classes"] for i in train)
    background_val = sum(not samples[i]["classes"] for i in val)

    summary = [
        f"随机种子：{RANDOM_SEED}",
        f"总图片：{len(samples)}",
        f"train：{len(train)} ({len(train) / len(samples):.2%})",
        f"val：{len(val)} ({len(val) / len(samples):.2%})",
        f"背景图片：total={background_total}, train={background_train}, val={background_val}",
        f"数据中实际出现类别：{len(existing_classes)}/{NUM_CLASSES}",
        f"train类别覆盖：{len(train_classes & existing_classes)}/{len(existing_classes)}",
        f"val类别覆盖：{len(val_classes & existing_classes)}/{len(existing_classes)}",
        f"val缺失类别：{sorted(existing_classes - val_classes)}",
        f"至少有2张图片但val仍缺失的类别：{sorted(splittable_classes - val_classes)}",
        f"强制保留在train的单样本图片数：{len(forced_train)}",
        f"train/val重复图片数：{len(train_set & val_set)}",
        "未划分图片数：0",
    ]
    (REPORT_DIR / "split_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))
    print(f"划分列表：{SPLIT_DIR}")
    print(f"逐类别统计：{REPORT_DIR / 'split_class_statistics.csv'}")


if __name__ == "__main__":
    main()
