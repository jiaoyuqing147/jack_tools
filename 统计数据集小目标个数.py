# -*- coding: utf-8 -*-
"""
TT100K 目标尺寸统计脚本（tiny / small / medium / large）

数据路径：
E:\DataSets\tt100k_2021\yolojack

YOLO标签格式：
class x_center y_center w h （归一化）

尺寸划分：
tiny   : area < 16^2
small  : 16^2 <= area < 32^2
medium : 32^2 <= area < 96^2
large  : area >= 96^2
"""

from pathlib import Path
from collections import Counter

# =========================
# ✅ 数据路径
# =========================
DATASET_ROOT = Path(r"E:\DataSets\tt100k_2021\yolojack")
LABELS_ROOT = DATASET_ROOT / "labels"

# =========================
# ✅ 输入图像尺寸
# 若你的标签是基于 640×640 输入统计，则保留 640
# =========================
IMG_SIZE = 640

# =========================
# ✅ 尺寸阈值（按面积）
# =========================
TINY_TH = 16 * 16
SMALL_TH = 32 * 32
MEDIUM_TH = 96 * 96


def get_size_type(area: float) -> str:
    """
    根据目标面积划分尺寸类别
    """
    if area < TINY_TH:
        return "tiny"
    elif area < SMALL_TH:
        return "small"
    elif area < MEDIUM_TH:
        return "medium"
    else:
        return "large"


def analyze_split(split: str):
    """
    统计单个数据划分（train / val / test）
    """
    label_dir = LABELS_ROOT / split
    size_counter = Counter()
    total_objects = 0

    if not label_dir.exists():
        print(f"[跳过] {split} 不存在: {label_dir}")
        return None

    txt_files = list(label_dir.glob("*.txt"))
    if not txt_files:
        print(f"[跳过] {split} 中没有找到 txt 标签文件: {label_dir}")
        return None

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                _, _, _, w, h = map(float, parts)
            except ValueError:
                continue

            w_px = w * IMG_SIZE
            h_px = h * IMG_SIZE
            area = w_px * h_px

            size_type = get_size_type(area)
            size_counter[size_type] += 1
            total_objects += 1

    print("\n" + "=" * 60)
    print(f"{split.upper()} SET")
    print("=" * 60)
    print(f"总目标数: {total_objects}")

    for k in ["tiny", "small", "medium", "large"]:
        count = size_counter[k]
        ratio = count / total_objects * 100 if total_objects > 0 else 0
        print(f"{k:>6}: {count:>8} ({ratio:6.2f}%)")

    return size_counter, total_objects


def main():
    print("=" * 60)
    print("TT100K tiny/small/medium/large 目标统计")
    print("=" * 60)
    print("数据路径:", DATASET_ROOT)
    print("标签路径:", LABELS_ROOT)
    print("图像尺寸:", IMG_SIZE)
    print("划分标准:")
    print(f"  tiny   : area < {16}² = {TINY_TH}")
    print(f"  small  : {16}² <= area < {32}²")
    print(f"  medium : {32}² <= area < {96}²")
    print(f"  large  : area >= {96}² = {MEDIUM_TH}")
    print("=" * 60)

    all_counter = Counter()
    all_total = 0

    for split in ["train", "val", "test"]:
        result = analyze_split(split)
        if result is None:
            continue

        counter, total = result
        all_counter += counter
        all_total += total

    print("\n" + "=" * 60)
    print("ALL SET (train + val + test)")
    print("=" * 60)
    print(f"总目标数: {all_total}")

    for k in ["tiny", "small", "medium", "large"]:
        count = all_counter[k]
        ratio = count / all_total * 100 if all_total > 0 else 0
        print(f"{k:>6}: {count:>8} ({ratio:6.2f}%)")


if __name__ == "__main__":
    main()