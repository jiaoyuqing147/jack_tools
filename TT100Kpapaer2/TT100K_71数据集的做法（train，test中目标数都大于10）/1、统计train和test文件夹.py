import json
from collections import Counter
from pathlib import Path

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

root = DATA_ROOT

train_dir = Path(root) / "train"
test_dir = Path(root) / "test"

json_file = Path(root) / "annotations_all.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

train_imgs = set()
test_imgs = set()

for p in train_dir.iterdir():
    if p.is_file():
        train_imgs.add(p.stem)

for p in test_dir.iterdir():
    if p.is_file():
        test_imgs.add(p.stem)

train_counter = Counter()
test_counter = Counter()

for img_name, img_info in data["imgs"].items():

    objects = img_info.get("objects", [])

    if img_name in train_imgs:

        for obj in objects:

            if "category" not in obj:
                continue

            cls = obj["category"]
            train_counter[cls] += 1

    elif img_name in test_imgs:

        for obj in objects:

            if "category" not in obj:
                continue

            cls = obj["category"]
            test_counter[cls] += 1

print("=" * 80)
print("TRAIN")
print("=" * 80)

for cls, num in sorted(train_counter.items(),
                       key=lambda x: x[1],
                       reverse=True):
    print(f"{cls:20s} {num}")

print()
print(f"Train Classes: {len(train_counter)}")
print()

print("=" * 80)
print("TEST")
print("=" * 80)

for cls, num in sorted(test_counter.items(),
                       key=lambda x: x[1],
                       reverse=True):
    print(f"{cls:20s} {num}")

print()
print(f"Test Classes: {len(test_counter)}")
print()

all_classes = set(train_counter.keys()) | set(test_counter.keys())

print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Total Classes : {len(all_classes)}")
print(f"Train Images  : {len(train_imgs)}")
print(f"Test Images   : {len(test_imgs)}")
train_cls = set(train_counter.keys())
test_cls = set(test_counter.keys())

print("Train only:")
print(sorted(train_cls - test_cls))

print()

print("Test only:")
print(sorted(test_cls - train_cls))