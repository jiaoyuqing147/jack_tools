"""
Controlled Low-light Generation for TT100K
==========================================

Simulate low illumination by reducing image brightness
with a moderate and reproducible degradation.

作者：
    Jiao Yuqing
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =====================================================
# 路径
# =====================================================

DATA_ROOT = Path(
    r"F:\DataSets\tt100k\yolojack"
)

INPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val"
)

OUTPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val_lowlight"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 参数
# =====================================================

BRIGHTNESS_FACTOR = 0.5   # 推荐 0.6，别太暗
GAMMA = 1.0               # 轻微 gamma，1.0 表示不做 gamma


def apply_gamma(img, gamma=1.1):
    if gamma == 1.0:
        return img

    img_float = img.astype(np.float32) / 255.0
    img_gamma = np.power(img_float, gamma)
    img_gamma = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)
    return img_gamma


# =====================================================
# 读取图片
# =====================================================

image_paths = sorted(
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.jpeg")) +
    list(INPUT_DIR.glob("*.png"))
)

print(f"Need process: {len(image_paths)} images")


# =====================================================
# 批量处理
# =====================================================

for img_path in tqdm(
        image_paths,
        desc="Generating Low-light Images"):

    save_path = (
        OUTPUT_DIR /
        img_path.name
    )

    img = cv2.imread(
        str(img_path)
    )

    if img is None:
        print(f"[WARNING] Failed to read: {img_path}")
        continue

    # 亮度降低
    low_img = img.astype(np.float32) * BRIGHTNESS_FACTOR
    low_img = np.clip(low_img, 0, 255).astype(np.uint8)

    # 轻微 gamma
    low_img = apply_gamma(
        low_img,
        gamma=GAMMA
    )

    cv2.imwrite(
        str(save_path),
        low_img
    )

print()
print("=" * 60)
print("Low-light Generation Finished")
print("=" * 60)