from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from collections import Counter

# ==========================
# 配置
# ==========================
ROOT = Path(r"E:\DataSets\tt100k_2021_paper2\tt100k_71")

SEED = 7 #要用4中得到的种子
VAL_RATIO = 0.1
NUM_CLASSES = 71

train_img_dir = ROOT / "images" / "train"
train_lbl_dir = ROOT / "labels" / "train"

val_img_dir = ROOT / "images" / "val"
val_lbl_dir = ROOT / "labels" / "val"


# ==========================
# 统计函数
# ==========================
def count_split(split):
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split

    images = list(img_dir.glob("*.*"))
    labels = list(lbl_dir.glob("*.txt"))

    cls_counter = Counter()
    box_num = 0

    for txt in labels:
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                cls_counter[cls_id] += 1
                box_num += 1

    print("=" * 70)
    print(split.upper())
    print("=" * 70)
    print(f"Images      : {len(images)}")
    print(f"Label files : {len(labels)}")
    print(f"Boxes       : {box_num}")
    print(f"Classes     : {len(cls_counter)}")
    print(f"Class IDs   : {sorted(cls_counter.keys())}")

    missing = sorted(set(range(NUM_CLASSES)) - set(cls_counter.keys()))
    if missing:
        print(f"Missing cls : {missing}")
    else:
        print("Missing cls : None")

    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": box_num,
        "classes": len(cls_counter),
        "missing": missing
    }


# ==========================
# 安全检查
# ==========================
if val_img_dir.exists() and any(val_img_dir.glob("*.*")):
    raise RuntimeError(
        f"{val_img_dir} 里面已经有图片。请先清空 val/images 和 val/labels，避免重复划分。"
    )

if val_lbl_dir.exists() and any(val_lbl_dir.glob("*.txt")):
    raise RuntimeError(
        f"{val_lbl_dir} 里面已经有标签。请先清空 val/images 和 val/labels，避免重复划分。"
    )

val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)


# ==========================
# 划分前统计
# ==========================
print("\nBefore split:")
before_train = count_split("train")


# ==========================
# 获取所有训练图片
# ==========================
all_images = sorted(
    [
        p for p in train_img_dir.glob("*.*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]
)

train_imgs, val_imgs = train_test_split(
    all_images,
    test_size=VAL_RATIO,
    random_state=SEED,
    shuffle=True
)

print("\nSplit plan:")
print(f"Total images : {len(all_images)}")
print(f"Train images : {len(train_imgs)}")
print(f"Val images   : {len(val_imgs)}")
print(f"Seed         : {SEED}")


# ==========================
# 移动验证集
# ==========================
moved_images = 0
moved_labels = 0
missing_labels = []

for img_path in val_imgs:
    label_path = train_lbl_dir / f"{img_path.stem}.txt"

    dst_img_path = val_img_dir / img_path.name
    dst_lbl_path = val_lbl_dir / f"{img_path.stem}.txt"

    shutil.move(str(img_path), str(dst_img_path))
    moved_images += 1

    if label_path.exists():
        shutil.move(str(label_path), str(dst_lbl_path))
        moved_labels += 1
    else:
        missing_labels.append(img_path.name)

print("\nMoved:")
print(f"Images moved : {moved_images}")
print(f"Labels moved : {moved_labels}")

if missing_labels:
    print("Missing label files:")
    for x in missing_labels:
        print(x)


# ==========================
# 划分后统计
# ==========================
print("\nAfter split:")
after_train = count_split("train")
after_val = count_split("val")
after_test = count_split("test")


# ==========================
# 总结检查
# ==========================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Train images: {after_train['images']}")
print(f"Val images  : {after_val['images']}")
print(f"Test images : {after_test['images']}")

print(f"Train boxes : {after_train['boxes']}")
print(f"Val boxes   : {after_val['boxes']}")
print(f"Test boxes  : {after_test['boxes']}")

train_val_boxes = after_train["boxes"] + after_val["boxes"]
val_box_ratio = after_val["boxes"] / train_val_boxes if train_val_boxes else 0

print(f"Val box ratio: {val_box_ratio:.4f}")

if (
    after_train["classes"] == NUM_CLASSES
    and after_val["classes"] == NUM_CLASSES
    and after_test["classes"] == NUM_CLASSES
):
    print("Class coverage: OK, train/val/test all contain 71 classes.")
else:
    print("Class coverage: WARNING, some split misses classes.")

print("Done.")