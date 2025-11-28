import os
import json
from pathlib import Path

from tqdm import tqdm
from PIL import Image  # pip install pillow

# ==============================
# 1. 路径设置：改成你的根目录
# ==============================
ROOT = r"E:\DataSets\MTSD\yolo54"  # 这里改成你的 yolo54 根目录

IMG_ROOT = Path(ROOT) / "images"
LBL_ROOT = Path(ROOT) / "labels"
ANNO_ROOT = Path(ROOT) / "annotations"
ANNO_ROOT.mkdir(exist_ok=True)

splits = ["train", "val", "test"]  # 有哪些划分就写哪些

# ==============================
# 2. 读取 classes.txt
# ==============================
classes_txt = LBL_ROOT / "classes.txt"
with open(classes_txt, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]

print(f"共读取到 {len(classes)} 个类别：")
for i, name in enumerate(classes):
    print(i, name)

categories = [{"id": i, "name": name} for i, name in enumerate(classes)]

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]


def find_image(label_path: Path, split: str) -> Path | None:
    """根据 label 文件名在 images/split 下面找对应图像."""
    stem = label_path.stem  # '0001' from '0001.txt'
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
        print(f"[警告] {lbl_dir} 不存在，跳过该划分")
        return

    label_files = sorted(p for p in lbl_dir.glob("*.txt"))
    print(f"{split} 中共找到 {len(label_files)} 个 txt 标注文件")

    for lbl_file in tqdm(label_files):
        img_path = find_image(lbl_file, split)
        if img_path is None:
            print(f"[警告] 找不到与 {lbl_file.name} 对应的图像，已跳过")
            continue

        # 读取图像尺寸
        with Image.open(img_path) as img:
            w, h = img.size

        images.append({
            "id": img_id,
            "file_name": str(img_path.relative_to(IMG_ROOT)),  # 例如 'train/0001.jpg'
            "width": w,
            "height": h
        })

        with open(lbl_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                print(f"[警告] {lbl_file} 中存在格式异常行: {line}")
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])

            # YOLO -> COCO
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            # 简单裁剪防止越界
            x = max(0, x)
            y = max(0, y)
            box_w = max(0, min(box_w, w - x))
            box_h = max(0, min(box_h, h - y))

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,  # 注意：和 classes 的索引一致
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

    print(f"{split} 转换完成，保存到: {out_file}")
    print(f"  images: {len(images)}, annotations: {len(annotations)}")


if __name__ == "__main__":
    for sp in splits:
        yolo2coco_for_split(sp)

    print("\n全部划分转换完成！")
