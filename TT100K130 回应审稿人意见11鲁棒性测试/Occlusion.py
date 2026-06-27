"""
BBox-guided Partial Occlusion Generation for TT100K
===================================================

Simulate partial occlusion by masking more than half of the annotated
traffic signs in each image. Each selected object is partially occluded
from a random side (left / right / top / bottom).

输入图像：
    F:\\DataSets\\tt100k\\yolojack\\images\\val

输入标签：
    F:\\DataSets\\tt100k\\yolojack\\labels\\val

输出图像：
    F:\\DataSets\\tt100k\\yolojack\\images\\val_occlusion

遮挡策略：
    每张图遮挡约 60% 的 GT 目标
    每个目标遮挡 bbox 的 45%
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
import math
import random
import pandas as pd
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

# 遮挡生成日志
LOG_CSV_PATH = (
    DATA_ROOT /
    "images" /
    "val_occlusion_generation_log.csv"
)


# =====================================================
# 参数
# =====================================================

SEED = 42

# 每张图遮挡超过一半目标
# 0.6 表示遮挡约 60% 的 GT 目标
OCCLUDED_OBJECT_RATIO = 0.6

# 每个被遮挡目标遮挡 bbox 的比例
# 0.45 表示遮挡目标框约 45%
OCCLUSION_RATIO = 0.45

# 至少遮挡几个目标
MIN_OCCLUDED_OBJECTS = 1

# 是否覆盖已经存在的遮挡图
# 重新生成遮挡数据集时必须 True，否则旧图不会更新
OVERWRITE_EXISTING = True

# 灰色遮挡块
MASK_COLOR = (114, 114, 114)

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

            cls_id = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            boxes.append(
                [
                    cls_id,
                    x_c,
                    y_c,
                    bw,
                    bh
                ]
            )

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


def apply_partial_occlusion(
        img,
        x1,
        y1,
        x2,
        y2,
        ratio=0.45):
    """
    对一个 bbox 的部分区域进行遮挡。
    遮挡方向随机：left / right / top / bottom
    """
    w = x2 - x1
    h = y2 - y1

    if w <= 1 or h <= 1:
        return img, None

    side = random.choice(
        [
            "left",
            "right",
            "top",
            "bottom"
        ]
    )

    if side == "left":
        occ_w = max(
            1,
            int(w * ratio)
        )

        ox1, oy1 = x1, y1
        ox2, oy2 = x1 + occ_w, y2

    elif side == "right":
        occ_w = max(
            1,
            int(w * ratio)
        )

        ox1, oy1 = x2 - occ_w, y1
        ox2, oy2 = x2, y2

    elif side == "top":
        occ_h = max(
            1,
            int(h * ratio)
        )

        ox1, oy1 = x1, y1
        ox2, oy2 = x2, y1 + occ_h

    else:
        occ_h = max(
            1,
            int(h * ratio)
        )

        ox1, oy1 = x1, y2 - occ_h
        ox2, oy2 = x2, y2

    cv2.rectangle(
        img,
        (ox1, oy1),
        (ox2, oy2),
        MASK_COLOR,
        thickness=-1
    )

    occ_info = {
        "side": side,
        "occ_x1": ox1,
        "occ_y1": oy1,
        "occ_x2": ox2,
        "occ_y2": oy2,
    }

    return img, occ_info


def decide_occlusion_count(num_boxes):
    """
    根据 GT 数量决定遮挡几个目标。
    目标：遮挡超过一半，默认约 60%。
    """
    if num_boxes <= 0:
        return 0

    num_occ = math.ceil(
        num_boxes * OCCLUDED_OBJECT_RATIO
    )

    num_occ = max(
        MIN_OCCLUDED_OBJECTS,
        num_occ
    )

    num_occ = min(
        num_occ,
        num_boxes
    )

    return num_occ


# =====================================================
# 批量处理
# =====================================================

log_rows = []

for img_path in tqdm(
        image_paths,
        desc="Generating Occlusion Images"):

    save_path = (
        OUTPUT_DIR /
        img_path.name
    )

    # 如果不覆盖已有图片，则跳过
    # 这次重新生成遮挡数据集，建议 OVERWRITE_EXISTING=True
    if save_path.exists() and not OVERWRITE_EXISTING:
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

    boxes = load_yolo_labels(
        label_path
    )

    # 没有标签就原图保存
    if len(boxes) == 0:
        cv2.imwrite(
            str(save_path),
            img
        )

        log_rows.append(
            {
                "image": img_path.stem,
                "image_path": str(img_path),
                "save_path": str(save_path),
                "gt_boxes": 0,
                "occluded_objects": 0,
                "occluded_ratio": 0.0,
                "selected_indices": "",
            }
        )

        continue

    # 根据 GT 数量决定遮挡几个目标
    num_occ = decide_occlusion_count(
        len(boxes)
    )

    selected_indices = random.sample(
        range(len(boxes)),
        num_occ
    )

    occluded = img.copy()

    selected_info = []

    for idx in selected_indices:
        x1, y1, x2, y2 = yolo_to_xyxy(
            boxes[idx],
            w,
            h
        )

        occluded, occ_info = apply_partial_occlusion(
            occluded,
            x1,
            y1,
            x2,
            y2,
            ratio=OCCLUSION_RATIO
        )

        if occ_info is not None:
            selected_info.append(
                {
                    "idx": idx,
                    "cls": boxes[idx][0],
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    **occ_info
                }
            )

    cv2.imwrite(
        str(save_path),
        occluded
    )

    log_rows.append(
        {
            "image": img_path.stem,
            "image_path": str(img_path),
            "save_path": str(save_path),
            "gt_boxes": len(boxes),
            "occluded_objects": num_occ,
            "occluded_ratio": num_occ / len(boxes),
            "selected_indices": ",".join(
                str(i) for i in selected_indices
            ),
            "selected_info": str(selected_info),
        }
    )


# =====================================================
# 保存日志
# =====================================================

df_log = pd.DataFrame(
    log_rows
)

df_log.to_csv(
    LOG_CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 60)
print("Occlusion Generation Finished")
print("=" * 60)
print(f"Input images : {INPUT_DIR}")
print(f"Input labels : {LABEL_DIR}")
print(f"Output images: {OUTPUT_DIR}")
print(f"Log CSV      : {LOG_CSV_PATH}")
print("=" * 60)