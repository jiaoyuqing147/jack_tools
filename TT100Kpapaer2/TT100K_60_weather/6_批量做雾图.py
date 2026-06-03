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
    r"E:\DataSets\tt100k_2021_paper2"
)

INPUT_DIR = (
    DATA_ROOT /
    "tt100k_60" /
    "images" /
    "train"
)

FOG_LIST = (
    DATA_ROOT /
    "tt100k_60_weather" /
    "fog.txt"
)

OUTPUT_DIR = (
    DATA_ROOT /
    "tt100k_60_weather" /
    "fog" /
    "images" /
    "train"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# 参数
# =====================================================

A = 245

beta = 2.0

# =====================================================
# 读取图片列表
# =====================================================

with open(
        FOG_LIST,
        "r",
        encoding="utf-8") as f:

    image_names = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"Need process: {len(image_names)} images"
)

# =====================================================
# 批量处理
# =====================================================

for stem in tqdm(
        image_names,
        desc="Generating Fog Images"):

    img_path = (
        INPUT_DIR /
        f"{stem}.jpg"
    )

    if not img_path.exists():
        continue

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