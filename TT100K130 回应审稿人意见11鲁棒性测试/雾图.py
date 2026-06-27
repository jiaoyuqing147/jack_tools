"""
Atmospheric Scattering Model for Fog Generation
==============================================

I(x) = J(x)t(x) + A(1-t(x))

t(x) = exp(-βd(x))

J(x) : 原图
I(x) : 雾图
A    : 大气光
β    : 雾浓度
d(x) : 距离

适用于：
    TT100K
    GTSDB
    Traffic Sign Detection

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

# 输入：原始 val 图像目录
INPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val"
)

# 输出：雾化后的 val 图像目录
OUTPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val_fog"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# 参数
# =====================================================

#这参数雾太浓，
# A = 245
# beta = 2.0

A = 225
beta = 0.9

# =====================================================
# 读取图片列表：不再使用 FOG_LIST
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
# 批量处理
# =====================================================

for img_path in tqdm(
        image_paths,
        desc="Generating Fog Images"):

    save_path = (
        OUTPUT_DIR /
        img_path.name
    )

    # 已存在直接跳过
    if save_path.exists():
        continue

    img = cv2.imread(
        str(img_path)
    )

    if img is None:
        print(f"[WARNING] Failed to read: {img_path}")
        continue

    img = img.astype(
        np.float32
    )

    h, w = img.shape[:2]

    # =================================================
    # 距离图
    # =================================================

    horizon = int(
        h * 0.45
    )

    distance = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for y in range(h):

        if y < horizon:

            distance[y, :] = 1.0

        else:

            distance[y, :] = (
                1.0 -
                (y - horizon)
                /
                (h - horizon)
            )

    # =================================================
    # Atmospheric Scattering
    # =================================================

    t = np.exp(
        -beta * distance
    )

    t = np.expand_dims(
        t,
        axis=2
    )

    foggy = (
        img * t
        +
        A * (1 - t)
    )

    foggy = np.clip(
        foggy,
        0,
        255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(save_path),
        foggy
    )

print()
print("=" * 60)
print("Finished")
print("=" * 60)