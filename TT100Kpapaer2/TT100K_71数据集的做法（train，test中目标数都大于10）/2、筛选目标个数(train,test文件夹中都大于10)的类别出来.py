import json
from pathlib import Path
from collections import Counter
DATA_ROOTS = [
    Path(r"E:\DataSets\tt100k_2021"),
    Path(r"/home/jiaoyuqing/datasets/tt100k_2021"),
    Path(r"D:\DataSets"),

]

DATA_ROOT = next(
    (p for p in DATA_ROOTS if p.exists()),
    None
)

if DATA_ROOT is None:
    raise RuntimeError("Cannot find DATA_ROOT")
# =====================================================
# 配置
# =====================================================
ROOT = DATA_ROOT

# 保留条件
MIN_TRAIN = 10
MIN_TEST = 10

# =====================================================
# 读取数据
# =====================================================
train_dir = Path(ROOT) / "train"
test_dir = Path(ROOT) / "test"
ann_file = Path(ROOT) / "annotations_all.json"

with open(ann_file, "r", encoding="utf-8") as f:
    data = json.load(f)

train_imgs = {p.stem for p in train_dir.iterdir() if p.is_file()}
test_imgs = {p.stem for p in test_dir.iterdir() if p.is_file()}

# =====================================================
# 统计类别
# =====================================================
train_counter = Counter()
test_counter = Counter()

for img_name, img_info in data["imgs"].items():

    objects = img_info.get("objects", [])

    for obj in objects:

        cls = obj.get("category")

        if cls is None:
            continue

        if img_name in train_imgs:
            train_counter[cls] += 1

        elif img_name in test_imgs:
            test_counter[cls] += 1

# =====================================================
# 汇总类别
# =====================================================
all_classes = sorted(
    set(train_counter.keys()) |
    set(test_counter.keys())
)

print("=" * 100)
print(
    f"Rule: KEEP if Train >= {MIN_TRAIN} "
    f"AND Test >= {MIN_TEST}"
)
print("=" * 100)

keep_classes = []
remove_classes = []

print(
    f"{'Class':20s}"
    f"{'Train':>10s}"
    f"{'Test':>10s}"
    f"{'Total':>10s}"
    f"{'Status':>12s}"
)

print("-" * 100)

for cls in all_classes:

    train_num = train_counter[cls]
    test_num = test_counter[cls]
    total_num = train_num + test_num

    if train_num >= MIN_TRAIN and test_num >= MIN_TEST:
        keep_classes.append(cls)
        status = "KEEP"
    else:
        remove_classes.append(cls)
        status = "REMOVE"

    print(
        f"{cls:20s}"
        f"{train_num:10d}"
        f"{test_num:10d}"
        f"{total_num:10d}"
        f"{status:>12s}"
    )

# =====================================================
# 统计信息
# =====================================================
print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"Original Classes : {len(all_classes)}")
print(f"Keep Classes     : {len(keep_classes)}")
print(f"Remove Classes   : {len(remove_classes)}")

print("\nKEEP CLASSES:")
print(",".join(sorted(keep_classes)))

print("\nREMOVE CLASSES:")
print(",".join(sorted(remove_classes)))

# =====================================================
# 保存结果
# =====================================================
save_tag = f"train_ge{MIN_TRAIN}_test_ge{MIN_TEST}"

keep_file = f"keep_classes_{save_tag}.txt"
remove_file = f"remove_classes_{save_tag}.txt"

with open(keep_file, "w", encoding="utf-8") as f:
    for cls in sorted(keep_classes):
        f.write(cls + "\n")

with open(remove_file, "w", encoding="utf-8") as f:
    for cls in sorted(remove_classes):
        f.write(cls + "\n")

print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(keep_file)
print(remove_file)

print("\n")
print("=" * 100)
print("FINAL RESULT")
print("=" * 100)

print(
    f"TT100K filtered classes "
    f"(Train >= {MIN_TRAIN}, Test >= {MIN_TEST}) "
    f"= {len(keep_classes)}"
)