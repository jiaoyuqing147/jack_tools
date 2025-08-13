import os
from collections import Counter
from pathlib import Path

# ===== 配置路径 =====
label_dirs = {
    'train': r"E:\tt100k_2021\yolo110\labels\train",
    'test':  r"E:\tt100k_2021\yolo110\labels\test",
    'other': r"E:\tt100k_2021\yolo110\labels\other"
}

# ===== 统计函数 =====
def count_classes(label_dir):
    counter = Counter()
    total_labels = 0
    label_dir = Path(label_dir)
    if not label_dir.exists():
        print(f"❌ 路径不存在: {label_dir}")
        return counter

    for file in label_dir.rglob("*.txt"):
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    cls_id = int(float(parts[0]))  # YOLO 格式第一列是类别 ID
                    counter[cls_id] += 1
                    total_labels += 1
    return counter, total_labels

# ===== 主逻辑 =====
for subset, path in label_dirs.items():
    cls_counter, total = count_classes(path)
    print(f"\n📂 {subset} 数据集:")
    print(f"  总标注框数量: {total}")
    print(f"  类别分布 (类别ID: 数量):")
    for cls_id, count in sorted(cls_counter.items()):
        print(f"    {cls_id}: {count}")
