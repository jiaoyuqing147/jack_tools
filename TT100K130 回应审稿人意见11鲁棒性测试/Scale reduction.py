"""
Scale Reduction Generation for TT100K
=====================================

Simulate low-resolution / scale degradation by downsampling
the image and then resizing it back to the original size.

适用于：
    TT100K
    GTSDB
    Traffic Sign Detection

作者：
    Jiao Yuqing
"""

import cv2
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

# 输出：尺度缩小后的 val 图像目录
OUTPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val_scale"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# 参数
# =====================================================

SCALE_FACTOR = 0.3   # 先缩小到 50%，再恢复原尺寸

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
# 批量处理
# =====================================================

for img_path in tqdm(
        image_paths,
        desc="Generating Scale Reduction Images"):

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

    # 缩小尺寸
    small_w = max(
        1,
        int(w * SCALE_FACTOR)
    )

    small_h = max(
        1,
        int(h * SCALE_FACTOR)
    )

    small_img = cv2.resize(
        img,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA
    )

    # 再放回原尺寸
    restored_img = cv2.resize(
        small_img,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imwrite(
        str(save_path),
        restored_img
    )

print()
print("=" * 60)
print("Scale Reduction Generation Finished")
print("=" * 60)