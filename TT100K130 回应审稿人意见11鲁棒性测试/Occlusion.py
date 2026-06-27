"""
BBox-guided Partial Occlusion Generation for TT100K
===================================================

Simulate partial occlusion by masking 1–3 annotated traffic signs
in each image. Each selected object is partially occluded from a
random side (left / right / top / bottom).
输入图像：F:\DataSets\tt100k\yolojack\images\val
输入标签：F:\DataSets\tt100k\yolojack\labels\val
输出图像：F:\DataSets\tt100k\yolojack\images\val_occlusion
每张图随机遮挡 1–3 个目标
每个目标遮挡 bbox 的 40%
遮挡方向随机：left / right / top / bottom
标签不用改
适用于：
    TT100K
    GTSDB
    Traffic Sign Detection

作者：
    Jiao Yuqing
"""

import cv2
import random
from pathlib import Path
from tqdm import tqdm


# =====================================================
# 路径
# =====================================================

DATA_ROOT = Path(
    r"F:\DataSets\tt100k\yolojack"
)

# 输入图像目录
INPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val"
)

# 输入标签目录（YOLO格式）
LABEL_DIR = (
    DATA_ROOT /
    "labels" /
    "val"
)

# 输出图像目录
OUTPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val_occlusion"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# 参数
# =====================================================

SEED = 42

MIN_OCCLUDED_OBJECTS = 1
MAX_OCCLUDED_OBJECTS = 3

OCCLUSION_RATIO = 0.6   # 遮挡目标 bbox 的 40%

MASK_COLOR = (114, 114, 114)   # 灰色遮挡，更自然一些

random.seed(SEED)

# =====================================================
# 读取图片列表
# =====================================================

image_paths = sorted(
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg")) +
    list(INPUT_DIR.glob("*.png"))
)

print(
    f"Need process: {len(image_paths)} images"
)

# =====================================================
# 工具函数：读取YOLO标签
# =====================================================

def load_yolo_labels(label_path):
    """
    返回:
        boxes: list of [cls_id, x_center, y_center, w, h]
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            boxes.append([cls_id, x_c, y_c, bw, bh])

    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    """
    YOLO normalized bbox -> pixel xyxy
    """
    _, x_c, y_c, bw, bh = box

    x1 = int((x_c - bw / 2) * img_w)
    y1 = int((y_c - bh / 2) * img_h)
    x2 = int((x_c + bw / 2) * img_w)
    y2 = int((y_c + bh / 2) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


def apply_partial_occlusion(img, x1, y1, x2, y2, ratio=0.4):
    """
    对一个 bbox 的 40% 区域进行部分遮挡
    遮挡方向随机：left / right / top / bottom
    """
    w = x2 - x1
    h = y2 - y1

    if w <= 1 or h <= 1:
        return img

    side = random.choice(
        ["left", "right", "top", "bottom"]
    )

    if side == "left":
        occ_w = max(1, int(w * ratio))
        ox1, oy1 = x1, y1
        ox2, oy2 = x1 + occ_w, y2

    elif side == "right":
        occ_w = max(1, int(w * ratio))
        ox1, oy1 = x2 - occ_w, y1
        ox2, oy2 = x2, y2

    elif side == "top":
        occ_h = max(1, int(h * ratio))
        ox1, oy1 = x1, y1
        ox2, oy2 = x2, y1 + occ_h

    else:  # bottom
        occ_h = max(1, int(h * ratio))
        ox1, oy1 = x1, y2 - occ_h
        ox2, oy2 = x2, y2

    cv2.rectangle(
        img,
        (ox1, oy1),
        (ox2, oy2),
        MASK_COLOR,
        thickness=-1
    )

    return img


# =====================================================
# 批量处理
# =====================================================

for img_path in tqdm(
        image_paths,
        desc="Generating Occlusion Images"):

    save_path = (
        OUTPUT_DIR /
        img_path.name
    )

    # 已存在则跳过
    if save_path.exists():
        continue

    img = cv2.imread(
        str(img_path)
    )

    if img is None:
        print(f"[WARNING] Failed to read: {img_path}")
        continue

    h, w = img.shape[:2]

    # 找到对应标签文件
    label_path = (
        LABEL_DIR /
        f"{img_path.stem}.txt"
    )

    boxes = load_yolo_labels(label_path)

    # 没有标签就原图保存
    if len(boxes) == 0:
        cv2.imwrite(
            str(save_path),
            img
        )
        continue

    # 随机决定遮挡几个目标
    max_k = min(
        MAX_OCCLUDED_OBJECTS,
        len(boxes)
    )

    min_k = min(
        MIN_OCCLUDED_OBJECTS,
        max_k
    )

    num_occ = random.randint(
        min_k,
        max_k
    )

    selected_indices = random.sample(
        range(len(boxes)),
        num_occ
    )

    occluded = img.copy()

    for idx in selected_indices:
        x1, y1, x2, y2 = yolo_to_xyxy(
            boxes[idx],
            w,
            h
        )

        occluded = apply_partial_occlusion(
            occluded,
            x1,
            y1,
            x2,
            y2,
            ratio=OCCLUSION_RATIO
        )

    cv2.imwrite(
        str(save_path),
        occluded
    )

print()
print("=" * 60)
print("Occlusion Generation Finished")
print("=" * 60)