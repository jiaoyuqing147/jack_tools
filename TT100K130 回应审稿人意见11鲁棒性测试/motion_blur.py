"""
Motion Blur Generation for TT100K
=================================

Simulate motion blur using a linear blur kernel.

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

# 输出：运动模糊后的 val 图像目录
OUTPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val_motionblur"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# 参数
# =====================================================

KERNEL_SIZE = 15   # 建议 9 / 15 / 21
ANGLE = 0          # 0=水平模糊，可扩展为随机角度

# =====================================================
# 构造运动模糊卷积核
# =====================================================

def motion_blur_kernel(kernel_size=15, angle=0):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    # 水平线
    kernel[kernel_size // 2, :] = np.ones(kernel_size, dtype=np.float32)

    # 归一化
    kernel /= kernel_size

    # 如果需要旋转角度
    if angle != 0:
        center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        s = kernel.sum()
        if s != 0:
            kernel /= s

    return kernel


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
# 生成卷积核
# =====================================================

kernel = motion_blur_kernel(
    kernel_size=KERNEL_SIZE,
    angle=ANGLE
)

# =====================================================
# 批量处理
# =====================================================

for img_path in tqdm(
        image_paths,
        desc="Generating Motion Blur Images"):

    save_path = (
        OUTPUT_DIR /
        img_path.name
    )

    # 已存在则跳过
    if save_path.exists():
        continue

    img = cv2.imread(str(img_path))

    if img is None:
        print(f"[WARNING] Failed to read: {img_path}")
        continue

    blurred = cv2.filter2D(
        img,
        -1,
        kernel
    )

    cv2.imwrite(
        str(save_path),
        blurred
    )

print()
print("=" * 60)
print("Motion Blur Generation Finished")
print("=" * 60)