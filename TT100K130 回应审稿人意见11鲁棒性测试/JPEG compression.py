"""
JPEG Compression Generation for TT100K
======================================

Simulate image compression degradation by re-encoding images
with a lower JPEG quality factor.

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

# 输出：JPEG压缩后的 val 图像目录
OUTPUT_DIR = (
    DATA_ROOT /
    "images" /
    "val_jpeg"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# 参数
# =====================================================

JPEG_QUALITY = 10   # 建议 30 / 50 / 70，50属于中等压缩

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
        desc="Generating JPEG Compression Images"):

    # 建议统一保存为 jpg，文件名保持 stem 不变
    save_path = (
        OUTPUT_DIR /
        f"{img_path.stem}.jpg"
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

    cv2.imwrite(
        str(save_path),
        img,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            JPEG_QUALITY
        ]
    )

print()
print("=" * 60)
print("JPEG Compression Generation Finished")
print("=" * 60)