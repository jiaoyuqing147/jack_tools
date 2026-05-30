from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(r"E:\DataSets\tt100k_2021_paper2\tt100k_71")

train_img_dir = ROOT / "images" / "train"
train_lbl_dir = ROOT / "labels" / "train"

all_images = list(train_img_dir.glob("*.*"))

TARGET_CLASSES = 71

for seed in range(10000):

    train_imgs, val_imgs = train_test_split(
        all_images,
        test_size=0.1,
        random_state=seed,
        shuffle=True
    )

    classes = set()

    train_boxes = 0
    val_boxes = 0

    val_set = set(val_imgs)

    for img_path in all_images:

        label_path = train_lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        with open(label_path, "r") as f:

            lines = f.readlines()

        num_boxes = len(lines)

        if img_path in val_set:

            val_boxes += num_boxes

            for line in lines:
                cls_id = int(line.split()[0])
                classes.add(cls_id)

        else:

            train_boxes += num_boxes

    ratio = val_boxes / (train_boxes + val_boxes)

    if len(classes) == TARGET_CLASSES and 0.08 <= ratio <= 0.12:

        print("Found!")
        print("Seed =", seed)
        print("Classes =", len(classes))
        print("Train Boxes =", train_boxes)
        print("Val Boxes =", val_boxes)
        print("Val Ratio =", ratio)

        break