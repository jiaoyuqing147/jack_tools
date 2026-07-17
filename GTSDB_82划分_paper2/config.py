from pathlib import Path

# ==================== 只需要优先修改这里 ====================
# 原始 GTSDB 图片目录：目录中可以是 .ppm/.png/.jpg 图片
SOURCE_IMAGE_DIR = Path(r"D:\Jiao\dataset\GTSDB\FullIJCNN2013")

# GTSDB 原始标注文件
GT_TXT_PATH = SOURCE_IMAGE_DIR / "gt.txt"

# 中间处理结果目录：统一 PNG 图片、YOLO 标签、划分列表和统计报告
WORK_DIR = Path(r"D:\Jiao\dataset\GTSDB\prepared_8_2")

# 最终可直接给 YOLO 使用的数据集目录
OUTPUT_DATASET_DIR = Path(r"D:\Jiao\dataset\GTSDB\GTSDB_YOLO_8_2")

# ==================== 划分参数 ====================
NUM_CLASSES = 43
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
RANDOM_SEED = 42

# 将所有图片统一转换为无损 PNG；建议保持 True
CONVERT_TO_PNG = True

# PNG 压缩级别只影响文件大小和保存速度，不影响图像内容，范围 0~9
PNG_COMPRESSION = 3

# 如果最终输出目录已经包含文件，是否允许清空其中的 images/labels/splits/reports。
# 为避免误删，默认 False；需要重新生成时建议换一个新目录。
CLEAN_OUTPUT = False

