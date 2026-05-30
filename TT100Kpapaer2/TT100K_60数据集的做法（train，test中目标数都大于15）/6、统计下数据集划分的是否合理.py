from pathlib import Path

# 自动寻找数据集根目录
for root in [
    Path(r"E:\DataSets"),               # Windows电脑1
    Path(r"D:\DataSets"),               # Windows电脑2（如果有）
    Path("/home/jiaoyuqing/datasets"),  # Linux服务器
]:
    if root.exists():
        DATA_ROOT = root
        break
else:
    raise RuntimeError("Cannot find DATA_ROOT")

ROOT = DATA_ROOT / "tt100k_2021_paper2" / "tt100k_60"

splits = ["train", "val", "test"]

for split in splits:

    label_dir = ROOT / "labels" / split

    classes = set()

    total_boxes = 0

    for txt_file in label_dir.glob("*.txt"):

        with open(txt_file, "r", encoding="utf-8") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])

                classes.add(cls_id)

                total_boxes += 1

    print("=" * 60)
    print(split.upper())
    print("=" * 60)

    print("Num Classes :", len(classes))
    print("Num Boxes   :", total_boxes)

    print("Class IDs:")
    print(sorted(classes))
    print()