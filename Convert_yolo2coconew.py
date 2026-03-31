import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from PIL import Image

# ==============================
# 1. 根目录
# ==============================
ROOT = Path("/home/jiaoyuqing/bigspace/workspaceJack/datasets/GTSDB/yolo43")
IMG_ROOT = ROOT / "images"
LBL_ROOT = ROOT / "labels"
ANNO_ROOT = ROOT / "annotations"
ANNO_ROOT.mkdir(exist_ok=True)

SPLITS = ["train", "val", "test"]
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

# ==============================
# 2. 读取 classes.txt
# ==============================
classes_txt = LBL_ROOT / "classes.txt"
with open(classes_txt, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]

print(f"共读取到 {len(classes)} 个类别")

# COCO 类别ID建议从1开始
categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]


def find_image(label_path: Path, split: str) -> Optional[Path]:
    stem = label_path.stem
    img_dir = IMG_ROOT / split
    for ext in IMG_EXTS:
        img_path = img_dir / f"{stem}{ext}"
        if img_path.exists():
            return img_path
    return None


def yolo2coco_for_split(split: str):
    print(f"\n=== 处理划分: {split} ===")
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    lbl_dir = LBL_ROOT / split
    if not lbl_dir.exists():
        print(f"[警告] {lbl_dir} 不存在，跳过")
        return

    label_files = sorted(lbl_dir.glob("*.txt"))
    print(f"{split} 中共找到 {len(label_files)} 个标注文件")

    for lbl_file in tqdm(label_files, desc=f"Converting {split}"):
        img_path = find_image(lbl_file, split)
        if img_path is None:
            print(f"[警告] 找不到与 {lbl_file.name} 对应的图像，跳过")
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        # file_name 写成相对于 images 根目录的路径
        # 例如 train/0001.jpg
        images.append({
            "id": img_id,
            "file_name": str(img_path.relative_to(IMG_ROOT)),
            "width": w,
            "height": h
        })

        with open(lbl_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                print(f"[警告] {lbl_file} 存在异常行: {line}")
                continue

            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            # YOLO -> COCO
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            # 越界裁剪
            x = max(0, x)
            y = max(0, y)
            box_w = max(0, min(box_w, w - x))
            box_h = max(0, min(box_h, h - y))

            if box_w <= 0 or box_h <= 0:
                print(f"[警告] {lbl_file} 存在非法框: {line}")
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id + 1,   # 注意 +1
                "bbox": [x, y, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    out_file = ANNO_ROOT / f"{split}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[完成] 保存到: {out_file}")
    print(f"images: {len(images)}, annotations: {len(annotations)}")


if __name__ == "__main__":
    for split in SPLITS:
        yolo2coco_for_split(split)

    print("\n全部转换完成！")